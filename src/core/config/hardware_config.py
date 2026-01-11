"""
Hardware Manifest.

This module defines the declarative schema for hardware abstraction and 
execution policy negotiation. It is responsible for resolving compute devices,
enforcing determinism constraints, and exposing hardware-derived execution
parameters.

It acts as the Single Source of Truth (SSOT) for:
    * Device selection and accelerator validation (CPU / CUDA / MPS)
    * Reproducibility and deterministic execution policy
    * DataLoader parallelism constraints
    * Process-level synchronization primitives (lock files)

This manifest guarantees that all hardware-related decisions are validated,
explicit, and immutable for the lifetime of an experiment.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse
import tempfile
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
from pydantic import (
    BaseModel, Field, field_validator, ConfigDict
)

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .types import ProjectSlug
from ..environment import detect_best_device, get_num_workers

# =========================================================================== #
#                                MODEL CONFIGURATION                          #
# =========================================================================== #

class HardwareConfig(BaseModel):
    """
    Declarative manifest for hardware abstraction and execution policy.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # Hardware Configuration
    device: str = Field(
        default_factory=detect_best_device,
        description="Computing device (cpu, cuda, mps or auto)."
    )

    # Execution Policy
    project_name: ProjectSlug = "vision_experiment"
    allow_process_kill: bool = Field(
        default=True,
        description="Permission flag to terminate duplicate processes for environment cleanup."
    )

    # Internal state for policy resolution
    _reproducible_mode: bool = False
    
    @property
    def lock_file_path(self) -> Path:
        """
        Dynamically generates a cross-platform lock file location.

        This path is used for environment sanitization and to prevent 
        resource contention during concurrent experiment execution.
        """
        safe_name = self.project_name.replace("/", "_")
        return Path(tempfile.gettempdir()) / f"{safe_name}.lock"

    @property
    def support_amp(self) -> bool:
        """Determines if the current validated device supports Automatic Mixed Precision (AMP)."""
        return self.device.lower().startswith("cuda") or \
               self.device.lower().startswith("mps")

    @property
    def effective_num_workers(self) -> int:
        """
        Calculates the optimal number of DataLoader workers.
        Returns 0 if reproducibility is required to avoid non-deterministic
        multiprocessing behavior, otherwise returns the system-detected maximum.
        """
        if self._reproducible_mode:
            return 0
        return get_num_workers()

    @property
    def use_deterministic_algorithms(self) -> bool:
        """Flag indicating whether PyTorch should enforce bit-perfect deterministic algorithms."""
        return self._reproducible_mode



    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        """
        SSOT Validation: Ensures the requested device actually exists on this system.
        If the requested accelerator (cuda/mps) is unavailable, it self-corrects to 'cpu'.
        """
        if v == "auto":
            return detect_best_device()

        requested = v.lower()
        if "cuda" in requested and not torch.cuda.is_available():
            return "cpu"
        if "mps" in requested and not torch.backends.mps.is_available():
            return "cpu"
        return requested

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "HardwareConfig":
        """
        Factory method to map CLI arguments to the HardwareConfig schema.
        """
        schema_fields = cls.model_fields.keys()

        params = {
            k: getattr(args, k)
            for k in schema_fields
            if hasattr(args, k) and getattr(args, k) is not None
        }

        instance = cls(**params)

        repro_flag = getattr(args, "reproducible", False)
        object.__setattr__(instance, "_reproducible_mode", repro_flag)

        return instance
