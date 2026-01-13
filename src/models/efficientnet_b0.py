"""
EfficientNet Architecture Engine for 224x224 Images.

This module adapts the EfficientNet architecture, originally designed for 
ImageNet, for use with 224x224 resolution images. EfficientNet, a family of 
models that balances model depth, width, and resolution for better efficiency 
and accuracy, is ideal for large-scale medical image classification tasks.

This engine supports pre-trained weight loading and custom adaptation for 
small-scale medical datasets, leveraging the pre-trained model's capability 
to generalize while fine-tuning it for specific medical imaging tasks.

Key Architectural Features:
    * Efficient Scaling: Balances depth, width, and resolution to optimize 
      both performance and computational efficiency.
    * Pre-trained Knowledge: Utilizes pre-trained weights for transfer learning,
      helping to jump-start the training process on small datasets.
    * Adaptable Input: Customizes input channels (e.g., grayscale or RGB) 
      and adapts the first layer to handle different input data.
"""

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
import torch.nn as nn
from torchvision import models

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config

# =========================================================================== #
#                               MODEL DEFINITION                              #
# =========================================================================== #

def build_efficientnet_b0(
        device: torch.device,
        num_classes: int,
        in_channels: int,
        cfg: Config
    ) -> nn.Module:
    """
    Fine-tunes an EfficientNet-B0 model specifically for 224x224 resolution 
    input images, such as those from medical datasets.

    EfficientNet-B0 is a baseline model from the EfficientNet family, optimized 
    for both accuracy and computational efficiency. This builder adapts the 
    input layer to handle different input channels (e.g., RGB or grayscale) 
    and modifies the classification head to match the number of classes in the 
    target dataset.

    Args:
        device: Target hardware for tensor placement (CPU/CUDA/MPS).
        num_classes: Cardinality of the classification head (dataset classes).
        in_channels: The number of input channels (e.g., 1 for Grayscale, 3 for RGB).
        cfg: Global configuration containing weight-loading and training policies.

    Returns:
        nn.Module: An EfficientNet-B0 model adapted for the given dataset and hardware.
    """
    
    # 1. Initialize EfficientNet-B0 with conditional pre-training
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.model.pretrained else None
    model = models.efficientnet_b0(weights=weights)
    
    # Snapshot of original weights before layer substitution
    old_conv = model.features[0][0]

    # 2. Adapt the input channels of the first convolutional layer
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=32,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        bias=False
    )
    
    # 3. Knowledge Distillation via Weight Morphing (if pretrained)
    if cfg.model.pretrained:
        with torch.no_grad():
            w = old_conv.weight
            
            # If input is Grayscale, compress the 3 RGB channels into 1 
            # by averaging, simulating a brightness-preserving conversion.
            if in_channels == 1:
                w = w.mean(dim=1, keepdim=True)

            new_conv.weight.copy_(w)
    
    # Replace the entry layer with our customized version
    model.features[0][0] = new_conv
    
    # 4. Modify the final classification layer to match the number of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Atomic synchronization to the execution device
    model = model.to(device)

    return model
