"""
Test Suite for Experiment Tracking Module.

Tests cover the create_tracker factory, NoOpTracker interface,
and tracker integration points in trainer, evaluation, and optimization.
"""

from unittest.mock import MagicMock

import pytest

from orchard.tracking import NoOpTracker, create_tracker

# --- FACTORY TESTS ---


@pytest.mark.unit
def test_create_tracker_no_tracking_config():
    """create_tracker returns NoOpTracker when cfg has no tracking attribute."""
    cfg = MagicMock(spec=[])
    tracker = create_tracker(cfg)
    assert isinstance(tracker, NoOpTracker)


@pytest.mark.unit
def test_create_tracker_tracking_none():
    """create_tracker returns NoOpTracker when cfg.tracking is None."""
    cfg = MagicMock()
    cfg.tracking = None
    tracker = create_tracker(cfg)
    assert isinstance(tracker, NoOpTracker)


@pytest.mark.unit
def test_create_tracker_tracking_disabled():
    """create_tracker returns NoOpTracker when tracking is disabled."""
    cfg = MagicMock()
    cfg.tracking.enabled = False
    tracker = create_tracker(cfg)
    assert isinstance(tracker, NoOpTracker)


# --- NOOP TRACKER TESTS ---


@pytest.mark.unit
def test_noop_tracker_start_run():
    """NoOpTracker.start_run completes without error."""
    tracker = NoOpTracker()
    tracker.start_run(cfg=MagicMock(), run_name="test", tracking_uri="file:///tmp")


@pytest.mark.unit
def test_noop_tracker_log_epoch():
    """NoOpTracker.log_epoch completes without error."""
    tracker = NoOpTracker()
    tracker.log_epoch(epoch=1, train_loss=0.5, val_metrics={"loss": 0.3}, lr=0.01)


@pytest.mark.unit
def test_noop_tracker_log_test_metrics():
    """NoOpTracker.log_test_metrics completes without error."""
    tracker = NoOpTracker()
    tracker.log_test_metrics(test_acc=0.95, macro_f1=0.90)


@pytest.mark.unit
def test_noop_tracker_log_artifact(tmp_path):
    """NoOpTracker.log_artifact completes without error."""
    tracker = NoOpTracker()
    tracker.log_artifact(tmp_path / "fake.txt")


@pytest.mark.unit
def test_noop_tracker_log_artifacts_dir(tmp_path):
    """NoOpTracker.log_artifacts_dir completes without error."""
    tracker = NoOpTracker()
    tracker.log_artifacts_dir(tmp_path)


@pytest.mark.unit
def test_noop_tracker_optuna_trial():
    """NoOpTracker nested trial methods complete without error."""
    tracker = NoOpTracker()
    tracker.start_optuna_trial(trial_number=0, params={"lr": 0.01})
    tracker.end_optuna_trial(best_metric=0.95)


@pytest.mark.unit
def test_noop_tracker_end_run():
    """NoOpTracker.end_run completes without error."""
    tracker = NoOpTracker()
    tracker.end_run()


@pytest.mark.unit
def test_noop_tracker_context_manager():
    """NoOpTracker works as a context manager."""
    with NoOpTracker() as tracker:
        assert isinstance(tracker, NoOpTracker)


@pytest.mark.unit
def test_create_tracker_mlflow_not_installed():
    """create_tracker returns NoOpTracker with warning when mlflow is missing."""
    from unittest.mock import patch

    cfg = MagicMock()
    cfg.tracking.enabled = True
    cfg.tracking.experiment_name = "test"

    with patch("orchard.tracking.tracker._MLFLOW_AVAILABLE", False):
        tracker = create_tracker(cfg)

    assert isinstance(tracker, NoOpTracker)


@pytest.mark.unit
def test_create_tracker_mlflow_available():
    """create_tracker returns MLflowTracker when mlflow is available."""
    from unittest.mock import patch

    from orchard.tracking.tracker import MLflowTracker

    cfg = MagicMock()
    cfg.tracking.enabled = True
    cfg.tracking.experiment_name = "my_experiment"

    with patch("orchard.tracking.tracker._MLFLOW_AVAILABLE", True):
        tracker = create_tracker(cfg)

    assert isinstance(tracker, MLflowTracker)
    assert tracker.experiment_name == "my_experiment"
