"""
Test Suite for OptunaConfig and InfrastructureManager.

Tests Optuna study configuration, early stopping params,
and infrastructure resource management.
"""
# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
from pathlib import Path

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import pytest
from pydantic import ValidationError

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core.config import (
    OptunaConfig, HardwareConfig, InfrastructureManager
)

# =========================================================================== #
#                    OPTUNA CONFIG: BASIC TESTS                               #
# =========================================================================== #

@pytest.mark.unit
def test_optuna_config_defaults():
    """Test OptunaConfig with default values."""
    config = OptunaConfig()
    
    assert config.study_name == "vision_optimization"
    assert config.n_trials == 50
    assert config.epochs == 15
    assert config.metric_name == "auc"
    assert config.direction == "maximize"
    assert config.sampler_type == "tpe"
    assert config.enable_early_stopping is True


@pytest.mark.unit
def test_optuna_config_early_stopping():
    """Test early stopping configuration."""
    config = OptunaConfig(
        enable_early_stopping=True,
        early_stopping_threshold=0.999,
        early_stopping_patience=2
    )
    
    assert config.enable_early_stopping is True
    assert config.early_stopping_threshold == 0.999
    assert config.early_stopping_patience == 2


@pytest.mark.unit
def test_optuna_config_from_args(optuna_args):
    """Test OptunaConfig.from_args()."""
    config = OptunaConfig.from_args(optuna_args)
    
    assert config.study_name == "test_study"
    assert config.n_trials == 10
    assert config.epochs == 15
    assert config.metric_name == "auc"


# =========================================================================== #
#                    OPTUNA CONFIG: VALIDATION                                #
# =========================================================================== #

@pytest.mark.unit
def test_invalid_metric_name_rejected():
    """Test invalid metric_name is rejected."""
    with pytest.raises(ValidationError, match="metric_name.*invalid"):
        OptunaConfig(metric_name="invalid_metric")


@pytest.mark.unit
def test_pruning_warmup_exceeds_epochs_rejected():
    """Test pruning_warmup_epochs >= epochs is rejected."""
    with pytest.raises(ValidationError, match="pruning_warmup"):
        OptunaConfig(epochs=10, pruning_warmup_epochs=10)


@pytest.mark.unit
def test_postgresql_without_storage_path_rejected():
    """Test PostgreSQL storage requires storage_path."""
    with pytest.raises(ValidationError, match="PostgreSQL.*storage_path"):
        OptunaConfig(storage_type="postgresql", storage_path=None)


# =========================================================================== #
#                    OPTUNA CONFIG: STORAGE URL                               #
# =========================================================================== #

@pytest.mark.unit
def test_get_storage_url_memory():
    """Test get_storage_url() for memory backend."""
    config = OptunaConfig(storage_type="memory")
    
    # Mock paths object
    class MockPaths:
        def get_db_path(self):
            return Path("/tmp/study.db")
    
    url = config.get_storage_url(MockPaths())
    assert url is None


@pytest.mark.unit
def test_get_storage_url_sqlite():
    """Test get_storage_url() for SQLite backend."""
    config = OptunaConfig(storage_type="sqlite")
    
    class MockPaths:
        def get_db_path(self):
            return Path("/tmp/test_study.db")
    
    url = config.get_storage_url(MockPaths())
    assert url.startswith("sqlite:///")
    assert "test_study.db" in url


# =========================================================================== #
#                INFRASTRUCTURE MANAGER: ENVIRONMENT PREP                     #
# =========================================================================== #

@pytest.mark.unit
def test_infrastructure_manager_creation():
    """Test InfrastructureManager can be instantiated."""
    manager = InfrastructureManager()
    
    assert manager is not None


@pytest.mark.integration
def test_prepare_environment_creates_lock(tmp_path):
    """Test prepare_environment creates lock file."""
    manager = InfrastructureManager()
    
    # Mock config with temp lock path
    class MockHardware:
        allow_process_kill = False  # Skip process cleanup
        lock_file_path = tmp_path / "test.lock"
    
    class MockConfig:
        hardware = MockHardware()
    
    config = MockConfig()
    
    # Should create lock file
    manager.prepare_environment(config)
    
    assert config.hardware.lock_file_path.exists()
    
    # Cleanup
    manager.release_resources(config)


@pytest.mark.integration
def test_release_resources_removes_lock(tmp_path):
    """Test release_resources removes lock file."""
    manager = InfrastructureManager()
    
    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"
    
    class MockConfig:
        hardware = MockHardware()
    
    config = MockConfig()
    
    # Create and release
    manager.prepare_environment(config)
    assert config.hardware.lock_file_path.exists()
    
    manager.release_resources(config)
    assert not config.hardware.lock_file_path.exists()


@pytest.mark.unit
def test_infrastructure_manager_frozen():
    """Test InfrastructureManager is frozen."""
    manager = InfrastructureManager()
    
    with pytest.raises(ValidationError):
        manager.new_field = "should_fail"


# =========================================================================== #
#                INFRASTRUCTURE MANAGER: CACHE FLUSHING                       #
# =========================================================================== #

@pytest.mark.unit
def test_flush_compute_cache_no_error():
    """Test _flush_compute_cache runs without error."""
    manager = InfrastructureManager()
    
    # Should not raise even if no GPU
    manager._flush_compute_cache()


# =========================================================================== #
#                    INTEGRATION: OPTUNA + INFRASTRUCTURE                     #
# =========================================================================== #

@pytest.mark.integration
def test_optuna_hardware_integration():
    """Test OptunaConfig works with HardwareConfig."""
    hw_config = HardwareConfig.for_optuna(device="cpu")
    optuna_config = OptunaConfig(n_trials=10)
    
    # Reproducible mode should be enabled
    assert hw_config.reproducible is True
    assert hw_config.effective_num_workers == 0
    
    # Optuna config should be valid
    assert optuna_config.n_trials == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])