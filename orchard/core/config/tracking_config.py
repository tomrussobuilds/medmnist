"""
Tracking Configuration.

Pydantic sub-config for experiment tracking settings (MLflow).
"""

from pydantic import BaseModel, Field


class TrackingConfig(BaseModel):
    """Configuration for MLflow experiment tracking.

    Controls whether MLflow logging is active and under which experiment
    name runs are grouped. When present in the YAML config, tracking
    is enabled by default.

    Attributes:
        enabled: Whether to activate MLflow logging for this run.
        experiment_name: MLflow experiment name (groups related runs).
    """

    enabled: bool = Field(default=True, description="Enable MLflow tracking")
    experiment_name: str = Field(
        default="visionforge",
        description="MLflow experiment name",
    )
