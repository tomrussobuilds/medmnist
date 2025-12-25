"""
Factories Package Initializer

This package centralizes the creation of model components, optimization 
strategies, and loss functions, decoupling the experiment configuration 
from the low-level PyTorch implementation.
"""

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .optim_factory import (
    get_optimizer, 
    get_scheduler, 
    get_criterion
)

# =========================================================================== #
#                                   Exports                                   #
# =========================================================================== #
__all__ = [
    "get_optimizer",
    "get_scheduler",
    "get_criterion"
]