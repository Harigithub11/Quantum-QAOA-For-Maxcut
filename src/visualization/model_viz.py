"""
Model Interpretability Visualizations
Grad-CAM, feature maps, saliency maps for understanding model decisions
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, List
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNNs.
    Shows which parts of the image the model focuses on.
    """

    def __init__(self, model: nn.Module, target_layer: str = 'layer4'):
        """
        Initialize Grad-CAM.

        Args:
            model: PyTorch model (ResNet18 in our case)
            target_layer: Layer name to visualize
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Get the target layer from ResNet18
        if hasattr(self.model, 'classical_extractor'):
            # For hybrid model
            resnet = self.model.classical_extractor.resnet
        else:
            # For standalone ResNet
            resnet = self.model

        target = dict([*resnet.named_modules()])[self.target_layer]
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Class Activation Map.

        Args:
            input_image: Input tensor (1, 1, 28, 28)
            target_class: Target class for visualization (None = predicted class)

        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        # Generate CAM
        gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        return cam

    def visualize(
        self,
        input_image: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        save_path: Optional[Path] = None,
        alpha: float = 0.5
    ) -> plt.Figure:
        """
        Visualize Grad-CAM overlayed on original image.

        Args:
            input_image: Input tensor
            original_image: Original image as numpy array (28, 28)
            target_class: Target class
            save_path: Path to save figure
            alpha: Overlay transparency

        Returns:
            Matplotlib figure
        """
        cam = self.generate_cam(input_image, target_class)

        # Resize CAM to match input image size
        cam_resized = cv2.resize(cam, (28, 28))

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Normalize original image
        img_normalized = ((original_image - original_image.min()) /
                         (original_image.max() - original_image.min()) * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)

        # Overlay
        overlayed = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')

        axes[2].imshow(overlayed)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


def visualize_feature_maps(
    model: nn.Module,
    input_image: torch.Tensor,
    layer_name: str = 'layer4',
    num_maps: int = 16,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualize feature maps from a specific layer.

    Args:
        model: PyTorch model
        input_image: Input tensor
        layer_name: Layer to visualize
        num_maps: Number of feature maps to show
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    activations = {}

    def hook_fn(module, input, output):
        activations[layer_name] = output.detach()

    # Register hook
    if hasattr(model, 'classical_extractor'):
        resnet = model.classical_extractor.resnet
    else:
        resnet = model

    layer = dict([*resnet.named_modules()])[layer_name]
    handle = layer.register_forward_hook(hook_fn)

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_image)

    # Get activations
    feature_maps = activations[layer_name][0].cpu().numpy()  # (C, H, W)

    # Remove hook
    handle.remove()

    # Plot
    num_maps = min(num_maps, feature_maps.shape[0])
    rows = int(np.sqrt(num_maps))
    cols = int(np.ceil(num_maps / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if num_maps > 1 else [axes]

    for idx in range(num_maps):
        axes[idx].imshow(feature_maps[idx], cmap='viridis')
        axes[idx].set_title(f'Map {idx}')
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(num_maps, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Feature Maps from {layer_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_filters(
    model: nn.Module,
    layer_name: str = 'conv1',
    num_filters: int = 64,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualize convolutional filters.

    Args:
        model: PyTorch model
        layer_name: Layer name
        num_filters: Number of filters to show
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if hasattr(model, 'classical_extractor'):
        resnet = model.classical_extractor.resnet
    else:
        resnet = model

    # Get weights
    layer = dict([*resnet.named_modules()])[layer_name]
    if isinstance(layer, nn.Conv2d):
        weights = layer.weight.detach().cpu().numpy()  # (out_ch, in_ch, H, W)
    else:
        raise ValueError(f"Layer {layer_name} is not a Conv2d layer")

    num_filters = min(num_filters, weights.shape[0])
    rows = int(np.sqrt(num_filters))
    cols = int(np.ceil(num_filters / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten() if num_filters > 1 else [axes]

    for idx in range(num_filters):
        # Take first input channel for visualization
        filter_img = weights[idx, 0]

        axes[idx].imshow(filter_img, cmap='gray')
        axes[idx].set_title(f'F{idx}', fontsize=8)
        axes[idx].axis('off')

    for idx in range(num_filters, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Convolutional Filters from {layer_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_activation_statistics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot activation statistics across dataset.

    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    model.eval()

    activations_list = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # Get classical features
            if hasattr(model, 'classical_extractor'):
                features = model.classical_extractor(images)
            else:
                features = model(images)

            activations_list.append(features.cpu().numpy())

            if len(activations_list) >= 10:  # Limit for speed
                break

    activations = np.concatenate(activations_list, axis=0)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram
    axes[0, 0].hist(activations.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('Activation Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Activation Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    # Box plot per feature
    axes[0, 1].boxplot([activations[:, i] for i in range(min(10, activations.shape[1]))],
                        labels=[f'F{i}' for i in range(min(10, activations.shape[1]))])
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('Activation Value')
    axes[0, 1].set_title('Feature-wise Distribution')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Mean and std per feature
    means = activations.mean(axis=0)
    stds = activations.std(axis=0)
    feature_indices = np.arange(len(means))

    axes[1, 0].bar(feature_indices, means, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Mean Activation')
    axes[1, 0].set_title('Mean Activation per Feature')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    axes[1, 1].bar(feature_indices, stds, alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Std Deviation')
    axes[1, 1].set_title('Activation Std per Feature')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    print("Model visualization module ready!")
    print("✓ Grad-CAM implementation")
    print("✓ Feature map visualization")
    print("✓ Filter visualization")
    print("✓ Activation statistics")
