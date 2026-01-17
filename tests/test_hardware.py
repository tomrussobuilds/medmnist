"""
Test Suite for HardwareConfig.

Tests device resolution, reproducibility, num_workers logic,
and lock file path generation.
"""
# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
import argparse
import tempfile

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import pytest
import torch

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core.config import HardwareConfig

# =========================================================================== #
#                         UNIT TESTS: DEVICE RESOLUTION                       #
# =========================================================================== #

@pytest.mark.unit
def test_device_auto_resolves():
    """Test device='auto' resolves to best available."""
    config = HardwareConfig(device="auto")
    
    # Should resolve to cuda, mps, or cpu
    assert config.device in ("cpu", "cuda", "mps")


@pytest.mark.unit
def test_device_cpu_always_works():
    """Test device='cpu' always resolves successfully."""
    config = HardwareConfig(device="cpu")
    
    assert config.device == "cpu"


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_cuda_when_available():
    """Test device='cuda' resolves when CUDA available."""
    config = HardwareConfig(device="cuda")
    
    assert config.device == "cuda"


@pytest.mark.unit
@pytest.mark.skipif(torch.cuda.is_available(), reason="Test requires no CUDA")
def test_device_cuda_fallback_to_cpu():
    """Test device='cuda' falls back to CPU when unavailable."""
    config = HardwareConfig(device="cuda")
    
    # Should fallback to CPU
    assert config.device == "cpu"


# =========================================================================== #
#                         UNIT TESTS: REPRODUCIBILITY                         #
# =========================================================================== #

@pytest.mark.unit
def test_reproducible_mode_disabled_by_default():
    """Test reproducible mode is False by default."""
    config = HardwareConfig()
    
    assert config.reproducible is False
    assert config.use_deterministic_algorithms is False


@pytest.mark.unit
def test_reproducible_mode_affects_num_workers():
    """Test reproducible mode forces num_workers=0."""
    config = HardwareConfig()
    config.reproducible = False
    
    # Should use system-detected workers
    assert config.effective_num_workers >= 0
    
    # Enable reproducible mode
    config.reproducible = True
    assert config.effective_num_workers == 0


@pytest.mark.unit
def test_for_optuna_factory_enables_reproducibility():
    """Test HardwareConfig.for_optuna() enables reproducible mode."""
    config = HardwareConfig.for_optuna(device="cpu")
    
    assert config.reproducible is True
    assert config.effective_num_workers == 0


# =========================================================================== #
#                         UNIT TESTS: AMP SUPPORT                             #
# =========================================================================== #

@pytest.mark.unit
def test_supports_amp_cpu_false():
    """Test CPU does not support AMP."""
    config = HardwareConfig(device="cpu")
    
    assert config.supports_amp is False


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_supports_amp_cuda_true():
    """Test CUDA supports AMP."""
    config = HardwareConfig(device="cuda")
    
    assert config.supports_amp is True


# =========================================================================== #
#                         UNIT TESTS: LOCK FILE PATH                          #
# =========================================================================== #

@pytest.mark.unit
def test_lock_file_path_in_temp_dir():
    """Test lock file is created in system temp directory."""
    config = HardwareConfig(project_name="test_project")
    
    lock_path = config.lock_file_path
    
    # Should be in temp directory
    assert str(lock_path).startswith(tempfile.gettempdir())
    assert lock_path.name == "test_project.lock"


@pytest.mark.unit
def test_lock_file_path_sanitizes_slashes():
    """Test lock file path sanitizes project name with slashes."""
    pytest.skip("ProjectSlug doesn't allow slashes by design")

    config = HardwareConfig(project_name="org/project")   
    lock_path = config.hardware.lock_file_path
    
    # Slashes should be replaced with underscores
    assert "/" not in lock_path.name
    assert "org_project.lock" in lock_path.name


# =========================================================================== #
#                         UNIT TESTS: FROM_ARGS FACTORY                       #
# =========================================================================== #

@pytest.mark.unit
def test_from_args_basic():
    """Test HardwareConfig.from_args() with basic arguments."""
    args = argparse.Namespace(
        device="cpu",
        project_name="test_exp",
        allow_process_kill=True
    )
    
    config = HardwareConfig.from_args(args)
    
    assert config.device == "cpu"
    assert config.project_name == "test_exp"
    assert config.allow_process_kill is True


@pytest.mark.unit
def test_from_args_ignores_none_values():
    """Test from_args ignores None values."""
    args = argparse.Namespace(
        device="cpu",
        project_name=None  # Should use default
    )
    
    config = HardwareConfig.from_args(args)
    
    assert config.device == "cpu"
    assert config.project_name == "vision_experiment"  # Default


# =========================================================================== #
#                         EDGE CASES & VALIDATION                             #
# =========================================================================== #

@pytest.mark.unit
def test_invalid_device_fallback():
    """Test invalid device type falls through validator."""
    # MPS on non-Mac should fallback to CPU
    config = HardwareConfig(device="mps")
    
    # Should either be mps (if on Mac) or cpu (fallback)
    assert config.device in ("mps", "cpu")


@pytest.mark.unit
def test_config_not_frozen():
    """Test HardwareConfig is NOT frozen (allows _reproducible_mode mutation)."""
    config = HardwareConfig()
    
    # Should be mutable (frozen=False)
    config.reproducible = True
    assert config.reproducible is True


@pytest.mark.unit
def test_project_name_validation():
    """Test project_name follows slug pattern."""
    # Valid slug
    config = HardwareConfig(project_name="valid-project_123")
    assert config.project_name == "valid-project_123"
    
    # Invalid should raise
    with pytest.raises(Exception):  # Pydantic validation error
        HardwareConfig(project_name="Invalid Project!")