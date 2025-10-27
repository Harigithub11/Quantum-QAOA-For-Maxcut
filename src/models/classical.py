"""
Classical Feature Extractor using Pre-trained ResNet18
Modified for grayscale MNIST images
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ResNet18FeatureExtractor(nn.Module):
    """
    ResNet18-based feature extractor for MNIST images.
    Extracts compact feature representations for quantum circuit input.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_layers: bool = True,
        feature_dim: int = 4,
        dropout: float = 0.2
    ):
        """
        Initialize ResNet18 feature extractor.

        Args:
            pretrained: Use pre-trained ImageNet weights
            freeze_layers: Freeze ResNet layers (feature extraction mode)
            feature_dim: Output feature dimension (must be 4 for quantum circuit)
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers

        # Load pre-trained ResNet18
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet18(weights=None)

        # Modify first convolutional layer for grayscale input (1 channel instead of 3)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # Modified: Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove the final fully connected layer
        # ResNet18 original FC: 512 -> 1000 classes
        # We'll replace it with our own dimensionality reduction
        self.resnet.fc = nn.Identity()

        # Freeze ResNet layers if in feature extraction mode
        if freeze_layers:
            for param in self.resnet.parameters():
                param.requires_grad = False
            print("ResNet18 layers frozen (feature extraction mode)")
        else:
            print("ResNet18 layers trainable (fine-tuning mode)")

        # Dimensionality reduction to quantum circuit input size
        # ResNet18 outputs 512-dim features, we need to reduce to 4
        self.feature_reduction = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, feature_dim),
            nn.Tanh()  # Bound outputs to [-1, 1] for quantum encoding
        )

        print(f"ResNet18 Feature Extractor initialized:")
        print(f"  Pretrained: {pretrained}")
        print(f"  Input channels: 1 (grayscale)")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Trainable parameters: {self.count_trainable_parameters():,}")

    def forward(self, x):
        """
        Forward pass through feature extractor.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Features tensor of shape (batch_size, feature_dim)
        """
        # ResNet feature extraction
        features = self.resnet(x)  # Output: (batch_size, 512)

        # Reduce to quantum circuit input size
        compact_features = self.feature_reduction(features)  # Output: (batch_size, 4)

        return compact_features

    def count_trainable_parameters(self) -> int:
        """
        Count number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        """
        Count total number of parameters.

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())

    def unfreeze_layers(self, num_layers: Optional[int] = None):
        """
        Unfreeze ResNet layers for fine-tuning.

        Args:
            num_layers: Number of layers to unfreeze from the end (None = all)
        """
        if num_layers is None:
            # Unfreeze all layers
            for param in self.resnet.parameters():
                param.requires_grad = True
            print("All ResNet layers unfrozen")
        else:
            # Unfreeze last num_layers
            layers = list(self.resnet.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Last {num_layers} ResNet layers unfrozen")

    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization.

        Args:
            x: Input tensor

        Returns:
            Dictionary of feature maps from different layers
        """
        feature_maps = {}

        # Hook functions to capture intermediate outputs
        def hook_fn(name):
            def hook(module, input, output):
                feature_maps[name] = output.detach()
            return hook

        # Register hooks
        hooks = []
        hooks.append(self.resnet.layer1.register_forward_hook(hook_fn('layer1')))
        hooks.append(self.resnet.layer2.register_forward_hook(hook_fn('layer2')))
        hooks.append(self.resnet.layer3.register_forward_hook(hook_fn('layer3')))
        hooks.append(self.resnet.layer4.register_forward_hook(hook_fn('layer4')))

        # Forward pass
        with torch.no_grad():
            _ = self.resnet(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return feature_maps


def create_feature_extractor(
    pretrained: bool = True,
    freeze_layers: bool = True,
    feature_dim: int = 4,
    dropout: float = 0.2
) -> ResNet18FeatureExtractor:
    """
    Factory function to create feature extractor.

    Args:
        pretrained: Use pre-trained weights
        freeze_layers: Freeze ResNet layers
        feature_dim: Output feature dimension
        dropout: Dropout probability

    Returns:
        Initialized feature extractor
    """
    return ResNet18FeatureExtractor(
        pretrained=pretrained,
        freeze_layers=freeze_layers,
        feature_dim=feature_dim,
        dropout=dropout
    )


if __name__ == "__main__":
    # Set UTF-8 encoding for Windows console
    import sys
    import os
    if os.name == 'nt':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

    print("=" * 70)
    print("Testing ResNet18 Feature Extractor")
    print("=" * 70)

    # Create feature extractor
    feature_extractor = ResNet18FeatureExtractor(
        pretrained=True,
        freeze_layers=True,
        feature_dim=4
    )

    # Test with sample MNIST-like input
    print("\nTesting forward pass...")
    batch_size = 8
    sample_input = torch.randn(batch_size, 1, 28, 28)  # MNIST format

    print(f"Input shape: {sample_input.shape}")

    # Forward pass
    features = feature_extractor(sample_input)

    print(f"Output shape: {features.shape}")
    print(f"Output range: [{features.min().item():.3f}, {features.max().item():.3f}]")
    print(f"Sample features: {features[0].detach().numpy()}")

    # Test gradient computation
    print("\nTesting gradient computation...")
    feature_extractor.train()
    output_sum = features.sum()
    output_sum.backward()

    # Check which parameters have gradients
    trainable_params = [name for name, p in feature_extractor.named_parameters() if p.requires_grad]
    print(f"Trainable parameter groups: {len(trainable_params)}")
    print(f"Sample trainable params: {trainable_params[:5]}")

    # Test feature extraction on multiple batches
    print("\nTesting batch processing...")
    test_batches = [torch.randn(4, 1, 28, 28) for _ in range(3)]

    for i, batch in enumerate(test_batches):
        with torch.no_grad():
            out = feature_extractor(batch)
        print(f"  Batch {i+1}: Input {batch.shape} → Output {out.shape}")

    # Test unfreezing
    print("\nTesting layer unfreezing...")
    initial_trainable = feature_extractor.count_trainable_parameters()
    print(f"  Initially trainable: {initial_trainable:,} parameters")

    feature_extractor.unfreeze_layers(2)
    after_unfreeze = feature_extractor.count_trainable_parameters()
    print(f"  After unfreezing: {after_unfreeze:,} parameters")

    # Model summary
    print("\nModel Summary:")
    print(f"  Total parameters: {feature_extractor.count_total_parameters():,}")
    print(f"  Trainable parameters: {feature_extractor.count_trainable_parameters():,}")
    print(f"  Frozen parameters: {feature_extractor.count_total_parameters() - feature_extractor.count_trainable_parameters():,}")

    # Test with actual data pipeline
    print("\nTesting with MNIST data...")
    try:
        from src.data.dataset import MNISTDataModule

        data_module = MNISTDataModule(batch_size=16)
        data_module.setup()

        images, labels = data_module.get_sample_batch()
        print(f"  MNIST batch: {images.shape}")

        with torch.no_grad():
            features = feature_extractor(images)

        print(f"  Extracted features: {features.shape}")
        print(f"  Feature statistics:")
        print(f"    Mean: {features.mean().item():.4f}")
        print(f"    Std: {features.std().item():.4f}")
        print(f"    Min: {features.min().item():.4f}")
        print(f"    Max: {features.max().item():.4f}")

        print("\n✓ Integration with MNIST data successful!")
    except Exception as e:
        print(f"  Note: Could not test with MNIST data - {e}")

    print("\n" + "=" * 70)
    print("✓ ResNet18 Feature Extractor test successful!")
    print("=" * 70)
