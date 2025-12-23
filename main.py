"""
Main Execution Script for MedMNIST Classification Pipeline

This orchestrator manages the lifecycle of a deep learning experiment, applying 
an adapted ResNet-18 architecture to various MedMNIST datasets (e.g., BloodMNIST). 

Key Pipeline Features:
1. Dynamic Configuration: Metadata-driven setup (mean/std, classes, channels) 
   leveraging a centralized Dataset Registry.
2. System Safety: Ensures environment reproducibility via seeding and prevents 
   resource conflicts through single-instance locking and process management.
3. Data Management: Handles automated loading, subset mocking for testing, 
   and robust PyTorch DataLoader creation with configurable augmentations.
4. Model Orchestration: Factory-based initialization of specialized architectures.
5. Training & Recovery: Executes standardized training loops with automated 
   checkpointing of the best model based on validation performance.
6. Comprehensive Evaluation: Performs final testing with Test-Time Augmentation (TTA), 
   generates diagnostic visualizations (Confusion Matrices, Loss Curves), and 
   exports structured performance reports in Excel format.
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import logging
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import torch

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from scripts.core import (
    Config, Logger, parse_args, set_seed, kill_duplicate_processes, get_cuda_name, 
    DATASET_REGISTRY, RunPaths, setup_static_directories, ensure_single_instance
)
from scripts.data_handler import (
    load_medmnist, get_dataloaders, show_sample_images, get_augmentations_description
)
from scripts.models import get_model
from scripts.trainer import ModelTrainer
from scripts.evaluation import run_final_evaluation

# =========================================================================== #
#                               MAIN EXECUTION
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

def main() -> None:
    """
    The main function that controls the entire training and evaluation flow.
    """
    
    # 1. Configuration Setup
    args = parse_args()
    
    # Initialize configuration from command-line arguments
    cfg = Config.from_args(args)
    
    # Initialize Seed
    set_seed(cfg.training.seed)

    # Setup base project structure
    setup_static_directories()

    # 2. Environment Initialization
    lock_path = Path("/tmp/medmnist_training.lock")

    # Initialize dynamic paths for the current run
    paths = RunPaths(
        dataset_slug=cfg.dataset.dataset_name,
        model_name=cfg.model_name,
        base_dir=cfg.system.output_dir
    )
    
    # Setup logger with run-specific file
    Logger.setup(
        name=paths.project_id,
        log_dir=paths.logs
    )
    run_logger = logging.getLogger(paths.project_id)
    
    ensure_single_instance(
        lock_file=lock_path,
        logger=run_logger
    )
    kill_duplicate_processes(
        logger=run_logger
    )

    device_str = cfg.system.device
    device = torch.device(device_str)
    run_logger.info(f"Execution Device: {device_str.upper()}")
    if args.device.lower() != device_str:
        run_logger.warning(
            f"Hardware Fallback: Requested '{args.device}', but using '{device_str}'"
        )
    
    run_logger.info(f"Run Directory initialized: {paths.root}")
    run_logger.info(
        f"Hyperparameters: LR={cfg.training.learning_rate:.4f}, Momentum={cfg.training.momentum:.2f}, "
        f"Batch={cfg.training.batch_size}, Epochs={cfg.training.epochs}, MixUp={cfg.training.mixup_alpha}, "
        f"TTA={'Enabled' if cfg.training.use_tta else 'Disabled'}"
    )

    # Retrieve dataset metadata from registry
    ds_meta = DATASET_REGISTRY[cfg.dataset.dataset_name.lower()]
    run_logger.info(f"Dataset selected: {cfg.dataset.dataset_name} with {cfg.dataset.num_classes} classes.")

    # 3. Data Loading and Preparation
    # 'data' is now a metadata container for Lazy Loading
    data = load_medmnist(ds_meta)

    # Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(data, cfg)
    
    # Optional: Visual check of samples (saved to run-specific figures directory)
    # Updated to use loader instead of raw X_train for Lazy Loading compatibility
    show_sample_images(
        loader=train_loader,
        classes=ds_meta.classes,
        save_path=paths.figures / "dataset_samples.png",
        cfg=cfg
    )

    # 4. Model Initialization (Factory Pattern)
    model = get_model(device=device, cfg=cfg)

    # 5. Training Execution
    run_logger.info("Starting training pipeline".center(60, "="))

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        output_dir=paths.models
    )
    best_path, train_losses, val_accuracies = trainer.train()

    # Load the best weights found during training
    model.load_state_dict(
        torch.load(
            best_path,
            map_location=device,
            weights_only=True
        )
    )
    run_logger.info(f"Loaded best checkpoint weights from: {best_path}")

    # 6. Final Evaluation (Metrics & Plots)
    aug_info = get_augmentations_description(cfg)

    # test_images and test_labels set to None to trigger Lazy extraction from loader
    macro_f1, test_acc = run_final_evaluation(
        model=model,
        test_loader=test_loader,
        test_images=None,
        test_labels=None,
        class_names=ds_meta.classes,
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        device=device,
        paths=paths,
        cfg=cfg,
        use_tta=cfg.training.use_tta,
        aug_info=aug_info
    )

    # Final Summary Logging
    run_logger.info(
        f"PIPELINE COMPLETED → "
        f"Test Acc: {test_acc:.4f} | "
        f"Macro F1: {macro_f1:.4f} | "
        f"Results saved in: {paths.root}"
    )


# =========================================================================== #
#                               ENTRY POINT
# =========================================================================== #

if __name__ == "__main__":
    main()