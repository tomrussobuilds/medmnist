"""
Configuration and Command-Line Interface Module

This module defines the training hyperparameters using Pydantic for validation
and type safety. It also provides the argument parsing logic for the 
command-line interface (CLI).
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import os
import argparse

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
from pydantic import BaseModel, Field, ConfigDict, field_validator

# =========================================================================== #
#                                HELPER FUNCTIONS
# =========================================================================== #

def _get_num_workers_config() -> int:
    """
    Calculates the default value for num_workers based on the environment.

    If DOCKER_REPRODUCIBILITY_MODE is set to '1' or 'TRUE', it returns 0
    to force single-thread execution for bit-per-bit determinism.
    
    Returns:
        int: The determined number of data loader workers (0 or 4).
    """
    is_docker_reproducible = os.environ.get("DOCKER_REPRODUCIBILITY_MODE", "0").upper() in ("1", "TRUE")
    return 0 if is_docker_reproducible else 4

# =========================================================================== #
#                                CONFIGURATION
# =========================================================================== #

class Config(BaseModel):
    """Configuration class for training hyperparameters using Pydantic validation."""
    model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            frozen=True
    )
    
    # Core Hyperparameters
    seed: int = 42
    batch_size: int = Field(default=128, gt=0)
    num_workers: int = Field(default_factory=_get_num_workers_config)
    epochs: int = Field(default=60, gt=0)
    patience: int = Field(default=15, ge=0)
    learning_rate: float = Field(default=0.008, gt=0)
    momentum: float = Field(default=0.9, ge=0.0, le=1.0)
    weight_decay: float = Field(default=5e-4, ge=0.0)
    mixup_alpha: float = Field(default=0.002, ge=0.0)
    use_tta: bool = True
    cosine_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Data Augmentation Parameters
    hflip: float = Field(default=0.5, ge=0.0, le=1.0)
    rotation_angle: int = Field(default=10, ge=0, le=180)
    jitter_val: float = Field(default=0.2, ge=0.0)

    model_name: str = "ResNet-18 Adapted"
    dataset_name: str = "BloodMNIST"

    # Dataset Metadata (to be populated from DATASET_REGISTRY)
    normalization_info : str =  "N/A"
    in_channels : int = 3
    num_classes : int = 8
    mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    std: tuple[float, ...] = (0.5, 0.5, 0.5)

    # Dataset Limits & Sampling
    max_samples: int | None = Field(default=20000, gt=0)
    use_weighted_sampler: bool = True

    @field_validator("num_workers")
    @classmethod
    def check_cpu_count(cls, v: int) -> int:
        cpu_count = os.cpu_count() or 1
        if v > cpu_count:
            return cpu_count
        return v

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        """
        Factory method to create a Config instance from CLI arguments
        and the central DATASET_REGISTRY.
        """
        from .dataset_metadata import DATASET_REGISTRY

        dataset_key = args.dataset.lower()
        if dataset_key not in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{args.dataset}' not found in DATASET_REGISTRY.")
        
        ds_meta = DATASET_REGISTRY[dataset_key]

        val_max = getattr(args, 'max_samples', 20000)
        final_max = val_max if val_max > 0 else None
        
        return cls(
            model_name=args.model_name,
            dataset_name=ds_meta.name,
            in_channels=ds_meta.in_channels,
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            mixup_alpha=args.mixup_alpha,
            use_tta=args.use_tta,
            hflip=args.hflip,
            rotation_angle=args.rotation_angle,
            jitter_val=args.jitter_val,
            max_samples=final_max,
            use_weighted_sampler=getattr(args, 'use_weighted_sampler', True),
            num_classes=len(ds_meta.classes),
            mean=ds_meta.mean,
            std=ds_meta.std,
            normalization_info=f"Mean={ds_meta.mean}, Std={ds_meta.std}",
        )

# =========================================================================== #
#                                ARGUMENT PARSING
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    """
    Configure and analyze command line arguments for the training script.

    Returns:
        argparse.Namespace: An object containing all parsed command line arguments.
    """
    from .dataset_metadata import DATASET_REGISTRY

    parser = argparse.ArgumentParser(
        description="MedMNIST training pipeline based on adapted ResNet-18.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    default_cfg = Config()

    # Group: Training Hyperparameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    
    train_group.add_argument(
        '--epochs',
        type=int,
        default=default_cfg.epochs
    )
    train_group.add_argument(
        '--batch_size',
        type=int,
        default=default_cfg.batch_size
    )
    train_group.add_argument(
        '--lr', '--learning_rate',
        type=float,
        default=default_cfg.learning_rate
    )
    train_group.add_argument(
        '--seed',
        type=int,
        default=default_cfg.seed
    )
    train_group.add_argument(
        '--patience',
        type=int,
        default=default_cfg.patience
    )
    train_group.add_argument(
        '--momentum',
        type=float,
        default=default_cfg.momentum
    )
    train_group.add_argument(
        '--weight_decay',
        type=float,
        default=default_cfg.weight_decay
    )
    train_group.add_argument(
        '--cosine_fraction',
        type=float,
        default=default_cfg.cosine_fraction,
        help="Fraction of total epochs to apply cosine annealing before switching to ReduceLROnPlateau."
    )
    # Group: Regularization & Augmentation
    aug_group = parser.add_argument_group("Regularization & Augmentation")
    
    aug_group.add_argument(
        '--mixup_alpha',
        type=float,
        default=default_cfg.mixup_alpha
    )
    aug_group.add_argument(
        '--no_tta',
        action='store_false',
        dest='use_tta',
        default=default_cfg.use_tta,
        help="Disable TTA during final evaluation."
    )
    aug_group.add_argument(
        '--hflip',
        type=float,
        default=default_cfg.hflip
    )
    aug_group.add_argument(
        '--rotation_angle',
        type=int,
        default=default_cfg.rotation_angle
    )
    aug_group.add_argument(
        '--jitter_val',
        type=float,
        default=default_cfg.jitter_val
    )

    # Group: Dataset Selection and Configuration
    dataset_group = parser.add_argument_group("Dataset Configuration")

    dataset_group.add_argument(
        '--dataset',
        type=str,
        default="bloodmnist",
        choices=DATASET_REGISTRY.keys(),
        help="Target MedMNIST dataset."
    )
    dataset_group.add_argument(
        '--max_samples',
        type=int,
        default=20000,
        help="Max training samples (None for full dataset)."
    )
    dataset_group.add_argument(
        '--balanced',
        action='store_true',
        dest='use_weighted_sampler',
        default=True,
        help="Use WeightedRandomSampler to handle class imbalance."
    )

    # Group: Model Selection
    model_group = parser.add_argument_group("Model Configuration")

    model_group.add_argument(
        '--model_name',
        type=str,
        default="ResNet-18 Adapted",
        help="Architecture identifier."
    )

    return parser.parse_args()