"""
Semantic Type Definitions & Validation Primitives.

This module acts as the foundational type-system for the configuration engine. 
It leverages Pydantic's Annotated types and Functional Validators to enforce 
domain-specific constraints (e.g., physical probability ranges, learning rate 
boundaries, and path integrity) before they reach the orchestration logic.

Core Responsibilities:
    * Path Sanitization: Automatic expansion and creation of experiment 
      directories via `_ensure_dir`.
    * Boundary Enforcement: Strict validation of hyperparameters (Learning Rates, 
      Probabilities, Smoothing Values) using LaTeX-style interval logic.
    * Reusability: Provides a centralized registry of Type Aliases used across 
      all sub-configuration modules (System, Training, Dataset).

By centralizing these definitions, the engine ensures that invalid states are 
caught at the 'edge' of the application (CLI/YAML parsing) rather than 
during active training execution.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from pathlib import Path
from typing import Annotated

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import (
    Field, AfterValidator
)

# =========================================================================== #
#                                VALIDATORS                                   #
# =========================================================================== #

def _ensure_dir(v: Path) -> Path:
    "Ensure paths are absolute and create folders if missing."
    v.mkdir(parents=True, exist_ok=True)
    return v.resolve()

# =========================================================================== #
#                                TYPE ALIASES                                 #
# =========================================================================== #

ValidatedPath = Annotated[Path, AfterValidator(_ensure_dir)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
SmoothingValue = Annotated[float, Field(ge=0.0, le=0.3)]
LearningRate = Annotated[float, Field(gt=1e-7, lt=1.0)]
Percentage = Annotated[float, Field(gt=0.0, le=1.0)]
Degrees = Annotated[int, Field(ge=0, le=180)]
