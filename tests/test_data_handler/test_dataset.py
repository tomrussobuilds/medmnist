"""
Pytest test suite for the VisionDataset class.

Covers dataset initialization, deterministic subsampling,
RGB vs grayscale handling, and __getitem__ behavior.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision import transforms

from orchard.data_handler.dataset import VisionDataset

_rng = np.random.default_rng(0)


# FIXTURES
@pytest.fixture
def rgb_npz(tmp_path: Path):
    """Creates a valid RGB MedMNIST-like NPZ."""
    path = tmp_path / "rgb.npz"
    np.savez(
        path,
        train_images=_rng.integers(0, 255, (20, 28, 28, 3), dtype=np.uint8),
        train_labels=np.arange(20),
        val_images=_rng.integers(0, 255, (10, 28, 28, 3), dtype=np.uint8),
        val_labels=np.arange(10),
        test_images=_rng.integers(0, 255, (10, 28, 28, 3), dtype=np.uint8),
        test_labels=np.arange(10),
    )
    return path


@pytest.fixture
def grayscale_npz(tmp_path: Path):
    """Creates a valid Grayscale MedMNIST-like NPZ."""
    path = tmp_path / "gray.npz"
    np.savez(
        path,
        train_images=_rng.integers(0, 255, (20, 28, 28), dtype=np.uint8),
        train_labels=np.arange(20),
        val_images=_rng.integers(0, 255, (10, 28, 28), dtype=np.uint8),
        val_labels=np.arange(10),
        test_images=_rng.integers(0, 255, (10, 28, 28), dtype=np.uint8),
        test_labels=np.arange(10),
    )
    return path


# TEST: Initialization Errors
def test_init_requires_existing_file(tmp_path):
    """Dataset initialization should fail if NPZ does not exist."""
    with pytest.raises(FileNotFoundError):
        VisionDataset(path=tmp_path / "missing.npz")


# TEST: Basic Loading
def test_len_matches_number_of_samples(rgb_npz):
    """__len__ should match number of loaded labels."""
    ds = VisionDataset(path=rgb_npz, split="train")
    assert len(ds) == 20


def test_getitem_returns_tensor_pair(rgb_npz):
    """__getitem__ should return (image, label) tensors."""
    ds = VisionDataset(path=rgb_npz, split="train")

    img, label = ds[0]

    assert isinstance(img, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long
    assert img.ndim == 3
    assert img.shape[0] == 3


# TEST: Grayscale Handling
def test_grayscale_images_are_expanded(grayscale_npz):
    """Grayscale datasets should be expanded to (H, W, 1)."""
    ds = VisionDataset(path=grayscale_npz, split="train")

    assert ds.images.ndim == 4
    assert ds.images.shape[-1] == 1

    img, _ = ds[0]
    assert img.shape[0] == 1


# TEST: Deterministic Subsampling
def test_max_samples_is_deterministic(rgb_npz):
    """Subsampling should be reproducible given the same seed."""
    ds1 = VisionDataset(path=rgb_npz, split="train", max_samples=5, seed=42)
    ds2 = VisionDataset(path=rgb_npz, split="train", max_samples=5, seed=42)

    assert len(ds1) == 5
    assert len(ds2) == 5
    assert np.array_equal(ds1.labels, ds2.labels)


def test_max_samples_smaller_than_dataset(rgb_npz):
    """max_samples should reduce dataset size."""
    ds = VisionDataset(path=rgb_npz, split="train", max_samples=7)
    assert len(ds) == 7


# TEST: Transform Application
def test_custom_transform_is_applied(rgb_npz):
    """Custom transforms should be applied to images."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    ds = VisionDataset(
        path=rgb_npz,
        split="train",
        transform=transform,
    )

    img, _ = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 3
    assert img.dtype == torch.float32


# TEST: Different Splits
@pytest.mark.parametrize("split,expected_len", [("train", 20), ("val", 10), ("test", 10)])
def test_dataset_splits(rgb_npz, split, expected_len):
    """Dataset should correctly load all supported splits."""
    ds = VisionDataset(path=rgb_npz, split=split)
    assert len(ds) == expected_len
