"""
PyTorch Dataset Definition Module

This module contains the custom Dataset class for MedMNIST, handling
the conversion from NumPy arrays to PyTorch tensors and applying 
image transformations for training and inference.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import Tuple
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# =========================================================================== #
#                              Internal Imports                               #
# =========================================================================== #
from scripts.core import Config

# =========================================================================== #
#                                DATASET CLASS                                #
# =========================================================================== #

class MedMNISTDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Enhanced PyTorch Dataset for MedMNIST data. Supports:
    - Lazy loading from disk (RAM efficient)
    - Subsampling with fixed seed (deterministic)
    - Automatic handling of RGB/Grayscale
    """
    def __init__(
            self,
            path: Path,
            split: str = "train",
            transform: transforms.Compose | None = None,
            max_samples: int | None = None,
            cfg: Config = Config
            ):
        """
        Args:
            path (Path): Path to the .npz file.
            split (str): One of 'train', 'val', or 'test'.
            transform (transforms.Compose | None): Torchvision transformations.
            max_samples (int | None): Limits the dataset size if it exceeds this value.
            cfg (Config): Global configuration for seeding.
        """
        self.path = path
        self.transform = transform
        self.split = split
        self.split_key = f"{split}_images"
        
        # Load labels and manage subsampling immediately
        with np.load(path) as data:
            # Note: label key is usually 'train_labels', yours had a space in the snippet
            label_key = f"{split}_labels"
            full_labels = data[label_key].ravel().astype(np.int64)
            total_available = len(full_labels)

            indices = np.arange(total_available)
            if max_samples and max_samples < total_available:
                rng = np.random.default_rng(cfg.seed)
                rng.shuffle(indices)
                self.indices = indices[:max_samples]
            else:
                self.indices = indices
            
            self.labels = full_labels[self.indices]

        # We set data_archive to None and open it only in __getitem__ or 
        # use a lazy property to ensure it's worker-safe.
        self.data_archive = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Lazy initialization for worker safety
        if self.data_archive is None:
            self.data_archive = np.load(self.path)

        actual_idx = self.indices[idx]
        img = self.data_archive[self.split_key][actual_idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        else:
            img = img.astype(np.float32) / 255.0
            if img.ndim == 2:
                img = torch.from_numpy(img).unsqueeze(0)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)

        return img, torch.tensor(label, dtype=torch.long)