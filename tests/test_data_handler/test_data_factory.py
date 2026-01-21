"""
Unit tests for DataLoaderFactory and related utilities.

Focus:
- DataLoaderFactory.build()
- WeightedRandomSampler
- _get_infrastructure_kwargs (Optuna, CUDA/MPS)
- LazyNPZDataset and create_temp_loader
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import torch

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core import DATASET_REGISTRY
from orchard.data_handler import DataLoaderFactory, LazyNPZDataset, create_temp_loader

# =========================================================================== #
#                          MOCK CONFIG AND METADATA                            #
# =========================================================================== #


@pytest.fixture
def mock_cfg():
    cfg = MagicMock()
    cfg.dataset.dataset_name = "mock_dataset"
    cfg.dataset.use_weighted_sampler = True
    cfg.dataset.max_samples = 10
    cfg.dataset.resolution = 28
    cfg.training.batch_size = 2
    cfg.num_workers = 0
    return cfg


@pytest.fixture
def mock_metadata():
    metadata = MagicMock()
    metadata.path = "/fake/path"
    return metadata


# =========================================================================== #
#                          DATA LOADER FACTORY TESTS                           #
# =========================================================================== #


def test_build_loaders_with_weighted_sampler(mock_cfg, mock_metadata):
    """Test DataLoaderFactory.build() with sampler and transforms."""
    with patch.dict(DATASET_REGISTRY, {"mock_dataset": MagicMock(in_channels=1)}):
        with patch(
            "orchard.data_handler.factory.get_pipeline_transforms",
            lambda cfg, meta: (lambda x: x, lambda x: x),
        ):

            class FakeDataset:
                def __init__(self, **kwargs):
                    self.labels = np.array([0, 1, 0, 1])

                def __len__(self):
                    return 4

            with patch("orchard.data_handler.factory.MedMNISTDataset", FakeDataset):
                factory = DataLoaderFactory(mock_cfg, mock_metadata)
                train, val, test = factory.build()

                # Check number of samples
                assert len(train.dataset) == 4
                assert len(val.dataset) == 4
                assert len(test.dataset) == 4

                # Check number of batches
                assert len(train) == 2
                assert len(val) == 2
                assert len(test) == 2

                # Check sampler is WeightedRandomSampler
                assert train.sampler is not None
                assert train.batch_size == mock_cfg.training.batch_size


def test_infra_kwargs_optuna(mock_cfg, mock_metadata):
    """Test _get_infrastructure_kwargs behavior in Optuna mode."""
    with patch.dict(DATASET_REGISTRY, {"mock_dataset": MagicMock(in_channels=1)}):
        factory = DataLoaderFactory(mock_cfg, mock_metadata)
        infra = factory._get_infrastructure_kwargs(is_optuna=True)
        assert infra["num_workers"] <= 6
        assert infra["persistent_workers"] is False


def test_infra_kwargs_pin_memory(monkeypatch, mock_cfg, mock_metadata):
    """Test that pin_memory is True if CUDA or MPS available."""
    with patch.dict(DATASET_REGISTRY, {"mock_dataset": MagicMock(in_channels=1)}):
        factory = DataLoaderFactory(mock_cfg, mock_metadata)
        monkeypatch.setattr(torch, "cuda", MagicMock(is_available=lambda: True))
        monkeypatch.setattr(torch.backends, "mps", MagicMock(is_available=lambda: False))

        infra = factory._get_infrastructure_kwargs()
        assert infra["pin_memory"] is True


# =========================================================================== #
#                             LAZY NPZ DATASET TESTS                           #
# =========================================================================== #


def test_lazy_npz_dataset():
    """Test LazyNPZDataset loads and returns tensors correctly."""
    # Create temporary npz
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dummy.npz"
        data = {
            "train_images": np.random.randint(0, 255, (5, 28, 28), dtype=np.uint8),
            "train_labels": np.random.randint(0, 2, (5, 1), dtype=np.int64),
        }
        np.savez(tmp_path, **data)

        dataset = LazyNPZDataset(tmp_path)
        assert len(dataset) == 5

        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape[0] == 1  # grayscale channel
        assert isinstance(label, int)


def test_create_temp_loader():
    """Test create_temp_loader returns a working DataLoader."""
    # Reuse temporary npz from LazyNPZDataset
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dummy.npz"
        data = {
            "train_images": np.random.randint(0, 255, (5, 28, 28), dtype=np.uint8),
            "train_labels": np.random.randint(0, 2, (5, 1), dtype=np.int64),
        }
        np.savez(tmp_path, **data)

        loader = create_temp_loader(tmp_path, batch_size=2)
        batch_imgs, batch_labels = next(iter(loader))
        assert batch_imgs.shape[0] <= 2
        assert batch_imgs.shape[1] == 1  # grayscale channel


# =========================================================================== #
#                            MAIN TEST RUNNER                                   #
# =========================================================================== #

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
