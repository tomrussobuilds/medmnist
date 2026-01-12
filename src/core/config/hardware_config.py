"""
Hardware Manifest.

This module defines the declarative schema for hardware abstraction and
execution policy negotiation. It is responsible for resolving the effective
compute device, enforcing determinism constraints, and deriving
hardware-dependent execution parameters.

It acts as the Single Source of Truth (SSOT) for:
    * Device selection policy with automatic accelerator resolution
      (CPU / CUDA / MPS via 'auto')
    * Reproducibility and deterministic execution guarantees
    * DataLoader parallelism constraints derived from execution policy
    * Process-level synchronization primitives (cross-platform lock files)

The manifest cleanly separates declarative configuration from internal
runtime state (e.g. reproducibility mode), ensuring that all hardware-related
decisions are validated, explicit, and immutable for the lifetime of an
experiment.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse
import tempfile
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
from pydantic import (
    BaseModel, Field, field_validator, ConfigDict
)

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .types import (
    ProjectSlug, DeviceType
)
from ..environment import (
    detect_best_device, get_num_workers
)

# =========================================================================== #
#                                MODEL CONFIGURATION                          #
# =========================================================================== #

class HardwareConfig(BaseModel):
    """
    Declarative manifest for hardware abstraction and execution policy.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # Hardware Configuration
    device: DeviceType = Field(
        default="auto",
        description="Computing device selection policy."
    )

    # Execution Policy
    project_name: ProjectSlug = "vision_experiment"
    allow_process_kill: bool = Field(
        default=True,
        description="Permission flag to terminate duplicate processes for environment cleanup."
    )

    # Internal, non serialized execution state
    _reproducible_mode: bool = False
    
    @property
    def lock_file_path(self) -> Path:
        """
        Dynamically generates a cross-platform lock file location.

        This path is used for environment sanitization and to prevent 
        resource contention during concurrent experiment execution.
        """
        safe_name = self.project_name.replace("/", "_")
        return Path(tempfile.gettempdir()) / f"{safe_name}.lock"

    @property
    def supports_amp(self) -> bool:
        """Whether validated device supports Automatic Mixed Precision (AMP)."""
        return self.device.lower().startswith("cuda") or \
               self.device.lower().startswith("mps")

    @property
    def effective_num_workers(self) -> int:
        """
        Calculates the optimal number of DataLoader workers.
        Returns 0 if reproducibility is required to avoid non-deterministic
        multiprocessing behavior, otherwise returns the system-detected maximum.
        """
        if self._reproducible_mode:
            return 0
        return get_num_workers()

    @property
    def use_deterministic_algorithms(self) -> bool:
        """Flag indicating whether PyTorch should enforce bit-perfect deterministic algorithms."""
        return self._reproducible_mode



    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: DeviceType) -> DeviceType:
        """
        SSOT Validation: Ensures the requested device actually exists on this system.
        If the requested accelerator (cuda/mps) is unavailable, it self-corrects to 'cpu'.
        """
        if v == "auto":
            return detect_best_device()

        requested = v.lower()

        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        
        return requested

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "HardwareConfig":
        """
        Factory method to map CLI arguments to the HardwareConfig schema.
        """
        schema_fields = cls.model_fields.keys()

        params = {
            k: getattr(args, k)
            for k in schema_fields
            if hasattr(args, k) and getattr(args, k) is not None
        }

        instance = cls(**params)

        repro_flag = getattr(args, "reproducible", False)
        object.__setattr__(instance, "_reproducible_mode", repro_flag)

        return instance
