"""
Health Check and Integrity Module

This script iterates through all registered MedMNIST datasets to:
1. Initialize the environment and security locks.
2. Download and verify MD5 checksums for each .npz file.
3. Validate internal keys and data consistency.
4. Generate visual samples to confirm correct mapping of labels/classes.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
from pathlib import Path

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from scripts.core import (
    Config, Logger, set_seed, get_device, 
    DATASET_REGISTRY, setup_static_directories, ensure_single_instance,
    kill_duplicate_processes
)
from scripts.data_handler import load_medmnist, show_sample_images

# =========================================================================== #
#                               HEALTH CHECK LOGIC                            #
# =========================================================================== #

def health_check() -> None:

    set_seed(42)
    setup_static_directories()

    log_dir = Path("outputs/health_checks")
    log_dir.mkdir(parents=True, exist_ok=True)
    Logger.setup(name="health_check", log_dir=log_dir)
    logger = logging.getLogger("health_check")

    lock_path = Path("/tmp/medmnist_health.lock")
    ensure_single_instance(lock_file=lock_path, logger=logger)
    kill_duplicate_processes(logger=logger)
    device = get_device(logger=logger)

    logger.info("="*60)
    logger.info("STARTING GLOBAL MEDMNIST HEALTH CHECK".center(60))
    logger.info("="*60)

    for key, ds_meta in DATASET_REGISTRY.items():
        logger.info(f"--- Checking Dataset: {ds_meta.display_name} ({key}) ---")
        
        try:

            num_classes_val = len(ds_meta.classes)

            temp_cfg = Config(
                model_name="HealthCheck-Probe",
                dataset_name=ds_meta.name,
                num_classes=len(ds_meta.classes),
                mean=ds_meta.mean,
                std=ds_meta.std,
                seed=42,
                batch_size=32,
                learning_rate=0.001,
                momentum=0.9,
                weight_decay=0.0,
                epochs=1,
                patience=1,
                mixup_alpha=0.0,
                use_tta=False,
                hflip=0.5,
                rotation_angle=0,
                jitter_val=0.0,
                normalization_info="",
            )

            data = load_medmnist(ds_meta)
            
            logger.info(f"Loaded successfully: Train={data.X_train.shape}, "
                        f"Val={data.X_val.shape}, Test={data.X_test.shape}")
            logger.info(f"Channels: {ds_meta.in_channels} | Classes: {num_classes_val}")

            sample_output_path = log_dir / f"samples_{ds_meta.name}.png"
            show_sample_images(
                images=data.X_train,
                labels=data.y_train,
                classes=ds_meta.classes,
                save_path=sample_output_path,
                cfg=temp_cfg
            )
            
            logger.info(f"Integrity check PASSED for {ds_meta.display_name}")

        except Exception as e:
            logger.error(f"Integrity check FAILED for {ds_meta.display_name}: {e}")
            continue

    logger.info("="*60)
    logger.info("GLOBAL HEALTH CHECK COMPLETED".center(60))
    logger.info("="*60)

# ========================================================================== #
#                                   ENTRY POINT                              #
# ========================================================================== # 
if __name__ == "__main__":
    health_check()