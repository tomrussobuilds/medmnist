"""Data Loader Orchestration Module.

This module provides the DataLoaderFactory class, which encapsulates the 
logic for building PyTorch DataLoaders. It handles dataset instantiation, 
class balancing via WeightedRandomSampler, and hardware-aware infrastructure 
setup (seeding, workers, memory pinning).
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
from typing import Dict, Optional, Tuple

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import DATASET_REGISTRY, Config, worker_init_fn
from .dataset import MedMNISTDataset
from .fetcher import MedMNISTData
from .transforms import get_pipeline_transforms

# =========================================================================== #
#                               DATALOADER FACTORY                            #
# =========================================================================== #

class DataLoaderFactory:
    """Orchestrates the creation of optimized PyTorch DataLoaders.

    This factory centralizes the configuration of training, validation, and 
    testing pipelines. It ensures that data transformations, class balancing, 
    and hardware settings are synchronized across all splits.

    Attributes:
        cfg (Config): Validated global configuration.
        metadata (MedMNISTData): Data path and raw format information.
        ds_meta (DatasetMetadata): Official MedMNIST registry specifications.
        logger (logging.Logger): Module-specific logger.
    """

    def __init__(self, cfg: Config, metadata: MedMNISTData):
        """Initializes the factory with environment and dataset metadata.

        Args:
            cfg: The global configuration object (Pydantic).
            metadata: Metadata from the data fetcher/downloader.
        """
        self.cfg = cfg
        self.metadata = metadata
        self.ds_meta = DATASET_REGISTRY[cfg.dataset.dataset_name]
        self.logger = logging.getLogger("medmnist_pipeline")

    def _get_transformation_pipelines(
            self
    ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Retrieves specialized vision pipelines.

        Returns:
            A tuple containing (train_transform, val_transform).
        """
        return get_pipeline_transforms(self.cfg, self.ds_meta)

    def _get_balancing_sampler(
            self,
            dataset: MedMNISTDataset
    ) -> Optional[WeightedRandomSampler]:
        """Calculates class weights and builds a WeightedRandomSampler.

        This method addresses class imbalance by assigning higher sampling 
        probabilities to under-represented classes.

        Args:
            dataset: The training dataset instance.

        Returns:
            A WeightedRandomSampler if enabled in config, otherwise None.
        """
        if not self.cfg.dataset.use_weighted_sampler:
            return None

        labels = dataset.labels.flatten()
        classes, counts = np.unique(labels, return_counts=True)
        
        # Inverse frequency balancing: weight = 1 / frequency
        class_weights = 1.0 / counts
        weight_map: Dict[int, float] = dict(zip(classes, class_weights))
        
        sample_weights = torch.tensor(
            [weight_map[int(label)] for label in labels], 
            dtype=torch.float
        )

        self.logger.info("Class balancing: WeightedRandomSampler generated.")
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    def _get_infrastructure_kwargs(self) -> dict:
        """Determines hardware and process-level parameters for DataLoaders.

        Returns:
            A dictionary of DataLoader arguments (num_workers, pin_memory, etc.).
        """
        num_workers = self.cfg.num_workers
        
        # Hardware acceleration: Pin memory for CUDA or MPS
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        
        return {
            "num_workers": num_workers,
            "pin_memory": has_cuda or has_mps,
            "worker_init_fn": worker_init_fn if num_workers > 0 else None,
            "persistent_workers": (num_workers > 0)
        }

    def build(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Constructs and returns the full suite of DataLoaders.

        This is the main entry point for the factory. It manages the 
        lifecycle of dataset splits and applies specific configurations 
        to each DataLoader.

        Returns:
            A tuple of (train_loader, val_loader, test_loader).
        """
        # 1. Setup transforms
        train_trans, val_trans = self._get_transformation_pipelines()

        # 2. Instantiate Dataset splits
        ds_params = {"path": self.metadata.path, "cfg": self.cfg}
        
        train_ds = MedMNISTDataset(
            **ds_params, split="train", 
            transform=train_trans, 
            max_samples=self.cfg.dataset.max_samples
        )

        # Proportional downsizing for validation/testing if max_samples is set
        sub_samples = None
        if self.cfg.dataset.max_samples:
            sub_samples = max(1, int(self.cfg.dataset.max_samples * 0.10))

        val_ds = MedMNISTDataset(
            **ds_params,
            split="val",
            transform=val_trans,
            max_samples=sub_samples
        )
        test_ds = MedMNISTDataset(
            **ds_params,
            split="test",
            transform=val_trans,
            max_samples=sub_samples
        )

        # 3. Resolve Sampler and Infrastructure
        sampler = self._get_balancing_sampler(train_ds)
        infra_kwargs = self._get_infrastructure_kwargs()

        # 4. Construct DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            **infra_kwargs
        )

        val_loader = DataLoader(
            val_ds, 
            batch_size=self.cfg.training.batch_size, 
            shuffle=False, 
            **infra_kwargs
        )
        
        test_loader = DataLoader(
            test_ds, 
            batch_size=self.cfg.training.batch_size, 
            shuffle=False, 
            **infra_kwargs
        )

        mode_str = "RGB" if self.ds_meta.in_channels == 3 else "Grayscale"
        self.logger.info(
            f"DataLoaders synchronized ({mode_str}) → "
            f"Train:[{len(train_ds)}] Val:[{len(val_ds)}] Test:[{len(test_ds)}]"
        )

        return train_loader, val_loader, test_loader
    

def get_dataloaders(metadata, cfg):
    """Convenience method for direct DataLoader retrieval."""
    factory = DataLoaderFactory(cfg, metadata)
    return factory.build()

# =========================================================================== #
#                               HEALTH UTILITIES                              #
# =========================================================================== #

def create_temp_loader(
        raw_data: np.lib.npyio.NpzFile,
        batch_size: int = 16
) -> DataLoader:
    """Direct conversion of raw NPZ arrays to DataLoader for health checks.

    Args:
        raw_data: The loaded NPZ file from numpy.
        batch_size: Samples per batch.

    Returns:
        A minimalistic DataLoader for early-stage pipeline verification.
    """
    images_t = torch.from_numpy(raw_data['train_images']).float() / 255.0
    
    if images_t.ndim == 3:
        images_t = images_t.unsqueeze(1)
    else:
        images_t = images_t.permute(0, 3, 1, 2)
        
    labels_t = torch.from_numpy(raw_data['train_labels']).long().squeeze()
    dataset = torch.utils.data.TensorDataset(images_t, labels_t)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)