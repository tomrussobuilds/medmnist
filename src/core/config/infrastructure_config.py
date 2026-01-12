"""
Infrastructure & Resource Lifecycle Management.

This module provides the operational bridge between the declarative configuration 
and the physical execution environment. It manages the 'clean-start' and 
'graceful-stop' sequences, ensuring that hardware resources are optimized 
and that concurrent experimental runs do not collide via filesystem-level locks.

Key Operational Tasks:
    * Process Sanitization: Guards against ghost processes and accidental 
      multi-process collisions in local environments.
    * Environment Locking: Implements a mutual exclusion (Mutex) strategy 
      to synchronize access to experimental outputs.
    * Resource De-allocation: Ensures GPU/MPS caches are flushed and temporary 
      system artifacts are purged upon exit.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import os
import logging
from typing import (
    Optional, Any, Protocol
)

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
from pydantic import BaseModel, ConfigDict

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from ..environment import (
    ensure_single_instance, release_single_instance, DuplicateProcessCleaner
)

# =========================================================================== #
#                             INFRASTRUCTURE MANAGER                          #
# =========================================================================== #

class HardwareAwareConfig(Protocol):
    """
    Structural contract for configurations exposing a hardware manifest.

    This protocol decouples infrastructure management from concrete
    configuration implementations, enabling type-safe access to
    hardware-related execution policies.
    """
    hardware: Any

class InfrastructureManager(BaseModel):
    """
    Operational executor for environment safeguarding and resource management.

    Ensures the execution environment is "clean" before a run starts and
    resources are released after the run, preventing collisions and leaks.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True
    )

    def prepare_environment(
        self,
        cfg: HardwareAwareConfig,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Prepares the environment for execution.

        Steps:
            1. Terminate duplicate or zombie processes if allowed.
            2. Acquire a filesystem lock to prevent concurrent runs.

        Args:
            cfg: Configuration exposing a hardware manifest.
            logger: Logger instance for status reporting.
        """
        log = logger or logging.getLogger("Infrastructure")

        # Process Sanitization
        if cfg.hardware.allow_process_kill:
            cleaner = DuplicateProcessCleaner()
            is_shared = any(
                env in os.environ
                for env in ["SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID"]
            )
            if not is_shared:
                num_zombies = cleaner.terminate_duplicates(logger=log)
                log.info(f" » Duplicate processes terminated: {num_zombies}.")
            else:
                log.debug(" » [SYS] Shared environment detected: skipping process kill.")

        # Concurrency Guarding
        ensure_single_instance(
            lock_file=cfg.hardware.lock_file_path,
            logger=log
        )
        log.info(f" » Lock acquired at {cfg.hardware.lock_file_path}")

    def release_resources(
        self,
        cfg: HardwareAwareConfig,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Releases system and hardware resources gracefully.

        Steps:
            1. Release filesystem lock.
            2. Flush hardware memory caches.

        Args:
            cfg: Configuration exposing a hardware manifest.
            logger: Logger instance for status reporting.
        """
        log = logger or logging.getLogger("Infrastructure")

        try:
            release_single_instance(cfg.hardware.lock_file_path)
            log.info(f" » Lock released at {cfg.hardware.lock_file_path}")
        except Exception as e:
            log.warning(f" » Failed to release lock file: {e}")

        self._flush_compute_cache(log=log)

    def _flush_compute_cache(self, log: Optional[logging.Logger] = None) -> None:
        """
        Clears GPU/MPS memory to prevent fragmentation across runs.
        """
        log = log or logging.getLogger("Infrastructure")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.debug(" » CUDA cache cleared.")
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                log.debug(" » MPS cache cleared.")
            except Exception:
                log.debug(" » MPS cache cleanup failed (non-fatal).")
