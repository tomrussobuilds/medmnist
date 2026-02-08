"""
Dataset Registry Orchestration & Metadata Resolution.

Bridges static dataset metadata with runtime execution requirements. Normalizes
datasets regardless of native format (Grayscale/RGB) to meet model architecture
input specifications. Supports multi-resolution (28x28, 224x224) with proper
YAML override while maintaining frozen immutability.

Key Responsibilities:
    * Adaptive normalization: Adjusts mean/std based on channel logic
    * Feature promotion: Automates grayscale-to-RGB for ImageNet weights
    * Resource budgeting: Enforces sampling limits and class balancing
    * Multi-resolution support: Resolves metadata by selected resolution
"""

import argparse
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..metadata import DatasetMetadata, DatasetRegistryWrapper
from ..paths import DATASET_DIR
from .types import ImageSize, PositiveInt, ValidatedPath


# DATASET CONFIGURATION
class DatasetConfig(BaseModel):
    """
    Validated manifest for dataset execution context.

    Bridges static registry metadata with runtime preferences. Resolves
    channel promotion and sampling policies with multi-resolution support.
    Auto-syncs img_size with resolution when not explicitly set.

    Attributes:
        name: Dataset identifier from registry (e.g., 'bloodmnist', 'organcmnist').
        metadata: DatasetMetadata object (excluded from serialization).
        data_root: Root directory containing dataset files.
        use_weighted_sampler: Enable class-balanced sampling for imbalanced datasets.
        max_samples: Maximum samples to load (None=all).
        img_size: Target square resolution for model input (auto-synced).
        force_rgb: Convert grayscale to RGB for pretrained ImageNet weights.
        resolution: Target resolution variant (28 or 224).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: Optional[str] = Field(
        default="bloodmnist", description="Dataset identifier (e.g., 'bloodmnist', 'organcmnist')"
    )
    metadata: Optional[DatasetMetadata] = Field(default=None, exclude=True)

    # Runtime parameters
    data_root: ValidatedPath = DATASET_DIR
    use_weighted_sampler: bool = True
    max_samples: Optional[PositiveInt] = Field(default=None)

    img_size: Optional[ImageSize] = Field(
        description="Target square resolution for model input",
        default=None,
    )
    force_rgb: bool = Field(
        default=True, description="Convert grayscale to RGB for ImageNet weights"
    )
    resolution: int = Field(default=28, description="Target dataset resolution (28 or 224)")

    @model_validator(mode="before")
    @classmethod
    def sync_img_size_with_resolution(cls, values):
        """
        Auto-sync img_size with resolution if not explicitly set.

        This runs BEFORE frozen instantiation, allowing us to modify values.

        Logic:
        1. If img_size is explicitly set in YAML/args → keep it
        2. If img_size is None/missing → use resolution
        3. If metadata exists → use metadata.native_resolution

        Args:
            values: Raw input dict before Pydantic validation

        Returns:
            Modified values dict with synced img_size
        """
        img_size = values.get("img_size")
        resolution = values.get("resolution", 28)
        metadata = values.get("metadata")

        if img_size is None:
            if metadata is not None:
                # Use metadata's native resolution
                img_size = metadata.native_resolution
            else:
                # Use resolution parameter
                img_size = resolution

            values["img_size"] = img_size

        return values

    # --- Properties ---

    @property
    def _ensure_metadata(self) -> DatasetMetadata:
        """
        Return metadata, lazily loading from registry if None.

        Uses object.__setattr__ to bypass frozen restriction for
        one-time initialization.

        Returns:
            DatasetMetadata object for this dataset.
        """
        if self.metadata is None:
            wrapper = DatasetRegistryWrapper(resolution=self.resolution)
            ds_name = self.name if self.name else list(wrapper.registry.keys())[0]
            metadata = wrapper.get_dataset(ds_name)
            object.__setattr__(self, "metadata", metadata)
        assert self.metadata is not None  # Guaranteed by the above
        return self.metadata

    @property
    def dataset_name(self) -> str:
        """
        Get dataset identifier from metadata.

        Returns:
            Dataset name string (e.g., 'bloodmnist').
        """
        return self._ensure_metadata.name

    @property
    def num_classes(self) -> int:
        """
        Get number of target classes.

        Returns:
            Integer count of classification classes.
        """
        return self._ensure_metadata.num_classes

    @property
    def in_channels(self) -> int:
        """
        Get native input channels from metadata.

        Returns:
            1 for grayscale datasets, 3 for RGB.
        """
        return self._ensure_metadata.in_channels

    @property
    def effective_in_channels(self) -> int:
        """
        Get actual channels the model receives after promotion.

        Returns:
            3 if force_rgb enabled, otherwise native in_channels.
        """
        return 3 if self.force_rgb else self.in_channels

    @property
    def mean(self) -> tuple[float, ...]:
        """
        Get channel-wise normalization mean.

        Expands single-channel mean to 3 channels if force_rgb is enabled
        on a grayscale dataset.

        Returns:
            Tuple of mean values per channel.
        """
        m = self._ensure_metadata.mean
        return (m[0],) * 3 if self.force_rgb and self.in_channels == 1 else m

    @property
    def std(self) -> tuple[float, ...]:
        """
        Get channel-wise normalization standard deviation.

        Expands single-channel std to 3 channels if force_rgb is enabled
        on a grayscale dataset.

        Returns:
            Tuple of std values per channel.
        """
        s = self._ensure_metadata.std
        return (s[0],) * 3 if self.force_rgb and self.in_channels == 1 else s

    @property
    def processing_mode(self) -> str:
        """
        Get description of channel processing mode.

        Returns:
            'NATIVE-RGB' for RGB datasets, 'RGB-PROMOTED' for grayscale
            with force_rgb, or 'NATIVE-GRAY' for grayscale without promotion.
        """
        if self.in_channels == 3:
            return "NATIVE-RGB"
        return "RGB-PROMOTED" if self.effective_in_channels == 3 else "NATIVE-GRAY"

    # --- Factory Methods ---

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DatasetConfig":
        """
        Create DatasetConfig from CLI arguments with resolution-aware metadata.

        Resolves dataset metadata from the registry based on the specified
        resolution (28 or 224), then constructs a fully configured instance
        with proper channel handling and sampling settings.

        Priority order for parameter resolution:
            1. YAML injection (if --config provided)
            2. CLI explicit arguments
            3. Inference from dataset registry
            4. Field defaults

        Args:
            args: Parsed argparse namespace containing:
                - dataset/name: Dataset identifier
                - resolution: Target resolution (28 or 224)
                - force_rgb: RGB promotion flag
                - max_samples: Sample limit
                - data_dir: Dataset root path
                - use_weighted_sampler: Class balancing flag

        Returns:
            Fully configured DatasetConfig with metadata loaded from registry.

        Raises:
            KeyError: If dataset not found in registry at specified resolution.
        """
        # 1. Determine dataset name
        dataset_name = getattr(args, "dataset", None) or getattr(args, "name", None) or "bloodmnist"

        # 2. Determine resolution
        resolution = getattr(args, "resolution", 28)

        # 3. Load metadata from registry
        wrapper = DatasetRegistryWrapper(resolution=resolution)
        try:
            resolved_metadata = wrapper.get_dataset(dataset_name)
        except KeyError:
            available = list(wrapper.registry.keys())
            raise KeyError(
                f"Dataset '{dataset_name}' not found at resolution {resolution}. "
                f"Available: {available}"
            )

        # 4. Resolve other params
        getattr(args, "pretrained", True)
        force_rgb_cli = getattr(args, "force_rgb", None)

        resolved_force_rgb = force_rgb_cli if force_rgb_cli is not None else True

        cli_max = getattr(args, "max_samples", None)
        resolved_max = None if (cli_max is not None and cli_max <= 0) else cli_max

        # Only pass explicit value if provided
        resolved_img_size = getattr(args, "img_size", None)

        # 5. Construct DatasetConfig
        return cls(
            name=dataset_name,
            data_root=Path(getattr(args, "data_dir", DATASET_DIR)),
            metadata=resolved_metadata,
            max_samples=resolved_max,
            use_weighted_sampler=getattr(args, "use_weighted_sampler", True),
            force_rgb=resolved_force_rgb,
            img_size=resolved_img_size,
            resolution=resolution,
        )
