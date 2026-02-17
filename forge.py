"""
Orchard ML: Unified ML Pipeline Entry Point.

Single entry point orchestrating the complete ML lifecycle:
    1. Hyperparameter Optimization (if optuna: section in config)
    2. Final Training with Best Parameters
    3. Model Export (if export: section in config)

All behavior is configuration-driven. No CLI flags for pipeline control.

Usage:
    # Use default recipe (ResNet-18 on PathMNIST 28x28)
    python forge.py

    # Full pipeline (tuning → training) with export
    python forge.py --config recipes/optuna_vit_tiny.yaml

    # Training only (no tuning, no export)
    python forge.py --config recipes/config_mini_cnn.yaml

    # Training + export (config has export: section)
    python forge.py --config recipes/config_resnet_18.yaml

Pipeline Logic:
    - If config contains `optuna:` section → runs optimization first
    - If config contains `export:` section → exports model after training
    - Pipeline duration tracked automatically by RootOrchestrator
"""

from orchard.core import Config, LogStyle, RootOrchestrator, log_pipeline_summary, parse_args
from orchard.core.paths import MLRUNS_DB
from orchard.pipeline import run_export_phase, run_optimization_phase, run_training_phase
from orchard.tracking import create_tracker


def main() -> None:
    """
    Main orchestrator for the unified forge pipeline.

    Executes ML lifecycle phases based on configuration:
        - Phase 1: Hyperparameter Optimization (if optuna config present)
        - Phase 2: Training with best/provided parameters
        - Phase 3: Model Export (if export config present)

    All timing is managed by RootOrchestrator's TimeTracker.
    """
    args = parse_args()
    cfg = Config.from_args(args)

    with RootOrchestrator(cfg) as orchestrator:
        run_logger = orchestrator.run_logger
        paths = orchestrator.paths

        # Experiment tracking (no-op if tracking section absent or mlflow not installed)
        tracker = create_tracker(cfg)
        tracking_uri = f"sqlite:///{MLRUNS_DB}"
        tracker.start_run(cfg=cfg, run_name=paths.run_id, tracking_uri=tracking_uri)

        training_cfg = cfg
        best_config_path = None

        try:
            # Phase 1: Optimization (if optuna config present)
            if cfg.optuna is not None:
                _, best_config_path = run_optimization_phase(orchestrator, tracker=tracker)

                # Load optimized config for training
                if best_config_path and best_config_path.exists():
                    args.config = str(best_config_path)
                    training_cfg = Config.from_args(args)
                    run_logger.info(f"Using optimized config: {best_config_path.name}")
            else:
                run_logger.info("Skipping optimization (no optuna config)")

            # Phase 2: Training
            best_model_path, _, _, _, macro_f1, test_acc = run_training_phase(
                orchestrator, cfg=training_cfg, tracker=tracker
            )

            # Phase 3: Export (if export config present)
            onnx_path = None
            if cfg.export is not None:
                onnx_path = run_export_phase(
                    orchestrator,
                    checkpoint_path=best_model_path,
                    cfg=training_cfg,
                    export_format=cfg.export.format,
                    opset_version=cfg.export.opset_version,
                )

            # Log final artifacts to MLflow
            tracker.log_artifacts_dir(paths.figures)
            tracker.log_artifact(paths.final_report_path)
            tracker.log_artifact(paths.get_config_path())

            # Pipeline Summary
            log_pipeline_summary(
                test_acc=test_acc,
                macro_f1=macro_f1,
                best_model_path=best_model_path,
                run_dir=paths.root,
                duration=orchestrator.time_tracker.elapsed_formatted,
                onnx_path=onnx_path,
                logger_instance=run_logger,
            )

        except KeyboardInterrupt:
            run_logger.warning(f"{LogStyle.WARNING} Interrupted by user.")
            raise SystemExit(1)

        except Exception as e:
            run_logger.error(f"{LogStyle.WARNING} Pipeline failed: {e}", exc_info=True)
            raise

        finally:
            tracker.end_run()


if __name__ == "__main__":
    main()
