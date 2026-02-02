"""
Export Configuration Schema.

Pydantic v2 schema defining model export parameters for ONNX and TorchScript.
Supports quantization, optimization, and validation settings.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .types import PositiveInt, ValidatedPath


# EXPORT CONFIGURATION
class ExportConfig(BaseModel):
    """
    Model export configuration for production deployment.

    Defines export format (ONNX/TorchScript), optimization level,
    quantization settings, and validation parameters.

    Example:
        >>> cfg = ExportConfig(
        ...     format="onnx",
        ...     opset_version=18,
        ...     quantize=True,
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ==================== Export Format ====================
    format: Literal["onnx", "torchscript", "both"] = Field(
        default="onnx", description="Export format"
    )

    output_path: Optional[ValidatedPath] = Field(
        default=None, description="Output path (auto-generated if None)"
    )

    # ==================== ONNX Settings ====================
    opset_version: PositiveInt = Field(
        default=18,
        description="ONNX opset version (18=latest, no conversion warnings). "
        "Lower versions may trigger fallback.",
    )

    dynamic_axes: bool = Field(
        default=True, description="Enable dynamic batch size (required for inference)"
    )

    do_constant_folding: bool = Field(
        default=True, description="Optimize constant operations at export time"
    )

    # ==================== TorchScript Settings ====================
    torchscript_method: Literal["trace", "script"] = Field(
        default="trace", description="TorchScript conversion method"
    )

    # ==================== Optimization ====================
    quantize: bool = Field(default=False, description="Apply INT8 quantization")

    quantization_backend: Literal["qnnpack", "fbgemm"] = Field(
        default="qnnpack", description="Quantization backend (qnnpack=mobile, fbgemm=x86)"
    )

    # ==================== Validation ====================
    validate_export: bool = Field(
        default=True, description="Validate exported model against PyTorch"
    )

    validation_samples: PositiveInt = Field(
        default=10, description="Number of samples for validation"
    )

    max_deviation: float = Field(
        default=1e-5,
        description="Maximum allowed output deviation between PyTorch and exported model",
    )

    @classmethod
    def from_args(cls, args) -> "ExportConfig":
        """
        Factory from CLI arguments.

        Args:
            args: Parsed argparse namespace

        Returns:
            ExportConfig instance
        """
        args_dict = vars(args)
        valid_fields = cls.model_fields.keys()
        params = {k: v for k, v in args_dict.items() if k in valid_fields and v is not None}
        return cls(**params)
