"""
ResNet-18 Adaptation Engine for Low-Resolution Medical Imaging.

This module specializes in "architectural surgery" for small-scale inputs (28x28). 
Standard ImageNet-centric models are designed for 224x224 images; applying them 
directly to MedMNIST results in aggressive spatial collapse. 

This engine preserves the Nyquist frequency of the input by modifying the early 
receptive fields and leveraging pre-trained knowledge through tensor interpolation 
(Weight Morphing), ensuring that the model starts with high-quality features 
despite the change in input geometry.
"""

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config

# =========================================================================== #
#                               MODEL DEFINITION                              #
# =========================================================================== #

def build_resnet18_adapted(
        device: torch.device,
        num_classes: int,
        in_channels: int,
        cfg: Config
    ) -> nn.Module:
    """
    Fine-tunes a ResNet-18 backbone specifically for the MedMNIST 28x28 format.

    The builder executes a transformation pipeline that reconciles the 
    mismatch between ImageNet pre-training (large, RGB) and MedMNIST 
    constraints (small, variable channels). It employs 'Selective Weight Transfer' 
    to maintain the benefit of pre-trained kernels while fitting a new 
    spatial entry point.

    Args:
        device: Target hardware for tensor placement (CPU/CUDA/MPS).
        num_classes: Cardinality of the classification head (dataset classes).
        in_channels: Source depth (1 for Grayscale datasets, 3 for RGB).
        cfg: Global configuration containing weight-loading and training policies.

    Returns:
        nn.Module: A model with 1:1 spatial resolution preservation in the first layer.
    """
    
    # 1. Initialize backbone with conditional pre-training
    # Validates against cfg.model.pretrained to ensure local/remote weight resolution.
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.model.pretrained else None
    model = models.resnet18(weights=weights)
    
    # Snapshot of original weights before layer substitution
    old_conv = model.conv1

    # 2. Re-engineer the input manifold
    # A 3x3 kernel with stride 1 ensures that the output feature map 
    # remains 28x28, preventing the "vanishing feature" problem of 7x7 stride-2.
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    # 3. Knowledge Distillation via Weight Morphing
    if cfg.model.pretrained:
        with torch.no_grad():
            w = old_conv.weight
            
            # Downsample 7x7 kernels to 3x3 using bicubic smoothing 
            # to preserve the peak intensity of the learned features.
            w = F.interpolate(w, size=(3,3), mode='bicubic', align_corners=True)

            # Conditional cross-modal adaptation:
            # If input is Grayscale, compress the 3 RGB channels into 1 
            # by averaging, simulating a brightness-preserving conversion.
            if in_channels == 1:
                w = w.mean(dim=1, keepdim=True)

            new_conv.weight.copy_(w)
    
    # Replace the entry layer with our spatially-optimized version
    model.conv1 = new_conv
    
    # Disable the initial MaxPool: standard ResNet would drop resolution 
    # to 7x7 here. Identity ensures the first Stage receives 28x28 features.
    model.maxpool = nn.Identity()
    
    # 4. Final Class Projection
    # Replaces the 1000-class ImageNet head with a task-specific linear layer.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Atomic synchronization to the execution device
    model = model.to(device)

    return model