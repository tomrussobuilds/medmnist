"""
Model Export Entry Point for VisionForge.

Converts trained PyTorch models to ONNX format for production deployment.

Usage:
    # Export Galaxy10 model
    python export.py --checkpoint outputs/run_xyz/models/best_efficientnetb0.pth \\
                     --dataset galaxy10 \\
                     --resolution 224 \\
                     --model_name efficientnet_b0 \\
                     --format onnx

    # Export MedMNIST model
    python export.py --checkpoint outputs/run_xyz/models/best_resnet18adapted.pth \\
                     --dataset bloodmnist \\
                     --resolution 28 \\
                     --model_name resnet_18_adapted \\
                     --format onnx

Important:
    The --dataset parameter provides architecture metadata (resolution, channels,
    num_classes) only - no data is loaded during export.

For detailed instructions, see: docs/guide/EXPORT.md
"""

from pathlib import Path

# Internal Imports
from orchard.core import Config, LogStyle, RootOrchestrator, parse_args
from orchard.export import export_to_onnx
from orchard.models.factory import get_model


# MAIN EXECUTION
def main() -> None:
    """
    Main orchestrator for model export execution.

    Coordinates checkpoint loading, format conversion, and validation.
    Utilizes RootOrchestrator for environment setup and resource management.

    Workflow:
        1. Parse CLI arguments (requires --checkpoint path)
        2. Build unified Config with export parameters
        3. Initialize orchestrator (device, logging)
        4. Load model architecture and weights
        5. Export to ONNX/TorchScript with optimizations
        6. Validate numerical consistency
        7. Report export statistics

    Raises:
        ValueError: If checkpoint path not provided or invalid
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: Export or validation failures
    """
    # Parse CLI arguments
    args = parse_args()

    # Validate checkpoint path
    checkpoint_path_arg = getattr(args, "checkpoint", None) or getattr(args, "resume", None)
    if not checkpoint_path_arg:
        raise ValueError(
            "Checkpoint path required for export. "
            "Use --checkpoint or --resume to specify the model checkpoint."
        )
    checkpoint_path = Path(checkpoint_path_arg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Validate export format
    if not hasattr(args, "format") or args.format is None:
        raise ValueError(
            "Export format required. Use --format to specify onnx, torchscript, or both."
        )

    # Build configuration
    cfg = Config.from_args(args)

    # Use orchestrator for environment setup
    with RootOrchestrator(cfg) as orchestrator:

        paths = orchestrator.paths
        logger = orchestrator.run_logger
        device = orchestrator.get_device()

        try:
            logger.info(f"{LogStyle.ARROW} Starting model export pipeline")
            logger.info(f"Checkpoint: {checkpoint_path}")
            logger.info(f"Format: {cfg.export.format}")
            logger.info(f"Device: {device}")

            # Build model architecture from config
            logger.info("Loading model architecture...")
            model = get_model(device=device, cfg=cfg)

            # Determine output path
            if cfg.export.output_path:
                output_path = cfg.export.output_path
            else:
                # Auto-generate output path in same directory as checkpoint
                stem = checkpoint_path.stem
                if cfg.export.format == "onnx":
                    output_path = checkpoint_path.parent / f"{stem}.onnx"
                elif cfg.export.format == "torchscript":
                    output_path = checkpoint_path.parent / f"{stem}.pt"
                else:  # both
                    output_path = checkpoint_path.parent / stem

            # Export to ONNX
            if cfg.export.format in ["onnx", "both"]:
                onnx_path = (
                    output_path if cfg.export.format == "onnx" else Path(str(output_path) + ".onnx")
                )

                logger.info(f"{LogStyle.ARROW} Exporting to ONNX format...")
                export_to_onnx(
                    model=model,
                    checkpoint_path=checkpoint_path,
                    output_path=onnx_path,
                    input_shape=(
                        cfg.dataset.effective_in_channels,
                        cfg.dataset.img_size,
                        cfg.dataset.img_size,
                    ),
                    opset_version=cfg.export.opset_version,
                    dynamic_axes=cfg.export.dynamic_axes,
                    do_constant_folding=cfg.export.do_constant_folding,
                    validate=cfg.export.validate_export,
                )

                logger.info(f"{LogStyle.SUCCESS} ONNX export complete: {onnx_path}")

            # Export to TorchScript
            if cfg.export.format in ["torchscript", "both"]:
                logger.info(f"{LogStyle.ARROW} Exporting to TorchScript format...")
                logger.warning("TorchScript export not yet implemented")
                # TODO: Implement torchscript_exporter.py

            logger.info(f"{LogStyle.SUCCESS} Export pipeline complete")
            logger.info(f"Output directory: {output_path.parent}")

        except KeyboardInterrupt:
            logger.warning(f"{LogStyle.WARNING} Export interrupted by user")

        except Exception as e:
            logger.error(f"{LogStyle.WARNING} Export failed: {e}", exc_info=True)
            raise

        finally:
            if "paths" in locals() and paths:
                logger.info(f"Export shutdown complete. Output: {paths.root}")


# ENTRY POINT
if __name__ == "__main__":
    main()
