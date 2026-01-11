"""
Telemetry & Filesystem Manifest.

This module defines the declarative schema for filesystem orchestration,
logging strategy, and experiment identity. It is responsible for path
sanitization, portable serialization, and experiment metadata.

It acts as the Single Source of Truth (SSOT) for:
    * Dataset and output directory resolution
    * Logging cadence and verbosity
    * Experiment identity and naming
    * Environment-agnostic manifest export

By centralizing telemetry concerns, the engine ensures that experiment
artifacts are traceable, portable, and free from host-specific leakage.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import (
    BaseModel, Field, field_validator, ConfigDict
)

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .types import (
    ValidatedPath, LogFrequency, LogLevel
)
from ..paths import PROJECT_ROOT

# =========================================================================== #
#                                MODEL CONFIGURATION                          #
# =========================================================================== #

class TelemetryConfig(BaseModel):
    """
    Declarative manifest for telemetry, logging, and filesystem strategy.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        json_encoders={
            Path: lambda v: str(v)
        }
    )

    # Filesystem Strategy
    data_dir: ValidatedPath = Field(default="./dataset")
    output_dir: ValidatedPath = Field(default="./outputs")

    # Telemetry
    save_model: bool = True
    log_interval: LogFrequency = Field(default=10)
    log_level: LogLevel = Field(default="INFO")

    def to_portable_dict(self) -> dict:
        """
        Converts the configuration instance into a portable dictionary.

        This method reconciles absolute system paths back to project-relative
        paths (e.g., converting '/home/user/project/dataset' to './dataset').
        It ensures that exported YAML/JSON manifests are environment-agnostic
        and do not leak local filesystem structures into logs or repositories.
        """
        data = self.model_dump()

        path_fields = ["data_dir", "output_dir"]

        for field in path_fields:
            full_path = Path(data[field])
            if full_path.is_relative_to(PROJECT_ROOT):
                relative_path = full_path.relative_to(PROJECT_ROOT)
                data[field] = f"./{relative_path}"
            else:
                data[field] = str(full_path)

        return data

    @field_validator("data_dir", "output_dir", mode="before")
    @classmethod
    def resolve_relative_paths(cls, v):
        """
        Ensures paths are always anchored to the PROJECT_ROOT.
        If 'v' is already absolute, it's kept as is (allowing external mounts).
        """
        path = Path(v)
        if not path.is_absolute():
            return (PROJECT_ROOT / path).resolve()
        return path.resolve()

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TelemetryConfig":
        """
        Factory method to map CLI arguments to the TelemetryConfig schema.
        """
        schema_fields = cls.model_fields.keys()

        params = {
            k: getattr(args, k)
            for k in schema_fields
            if hasattr(args, k) and getattr(args, k) is not None
        }

        return cls(**params)
