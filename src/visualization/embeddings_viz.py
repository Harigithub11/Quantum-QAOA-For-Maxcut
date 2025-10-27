"""
Feature Embeddings Visualization
t-SNE and UMAP for dimensionality reduction and visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm


def extract_features(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 1000,
    layer: str = 'classical'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from model for visualization.

    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device
        max_samples: Maximum samples to extract
        layer: Which layer to extract ('classical', 'quantum', or 'final')

    Returns:
        Tuple of (features, labels)
    """
    model.eval()

    features_list = []
    labels_list = []
    samples_collected = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            if samples_collected >= max_samples:
                break

            images = images.to(device)

            if layer == 'classical':
                # Extract classical features (ResNet18 output)
                if hasattr(model, 'classical_extractor'):
                    feats = model.classical_extractor(images)
                else:
                    feats = model(images)

            elif layer == 'quantum':
                # Extract quantum features
                if hasattr(model, 'forward_with_features'):
                    outputs = model.forward_with_features(images)
                    feats = outputs['quantum_output']
                else:
                    raise ValueError("Model doesn't support quantum feature extraction")

            elif layer == 'final':
                # Extract final layer features (before softmax)
                feats = model(images)

            else:
                raise ValueError(f"Unknown layer: {layer}")

            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

            samples_collected += len(images)

    features = np.concatenate(features_list, axis=0)[:max_samples]
    labels = np.concatenate(labels_list, axis=0)[:max_samples]

    return features, labels


def plot_tsne_2d(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[Path] = None,
    interactive: bool = True,
    perplexity: int = 30
) -> go.Figure:
    """
    Create 2D t-SNE visualization.

    Args:
        features: Feature array (N, D)
        labels: Labels array (N,)
        save_path: Path to save figure
        interactive: Use Plotly (True) or Matplotlib (False)
        perplexity: t-SNE perplexity parameter

    Returns:
        Plotly or Matplotlib figure
    """
    print(f"Running t-SNE on {len(features)} samples...")

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    embeddings = tsne.fit_transform(features)

    if interactive:
        # Plotly interactive version
        fig = px.scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            color=labels.astype(str),
            labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Digit'},
            title='t-SNE Visualization of Features',
            color_discrete_sequence=px.colors.qualitative.Set1
        )

        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(
            width=800,
            height=600,
            hovermode='closest'
        )

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path.with_suffix('.html')))
            fig.write_image(str(save_path))

        return fig

    else:
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=labels,
            cmap='tab10',
            s=20,
            alpha=0.7
        )

        plt.colorbar(scatter, label='Digit', ax=ax)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('t-SNE Visualization of Features')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


def plot_tsne_3d(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[Path] = None,
    perplexity: int = 30
) -> go.Figure:
    """
    Create 3D t-SNE visualization (Plotly only).

    Args:
        features: Feature array (N, D)
        labels: Labels array (N,)
        save_path: Path to save figure
        perplexity: t-SNE perplexity parameter

    Returns:
        Plotly figure
    """
    print(f"Running 3D t-SNE on {len(features)} samples...")

    # Apply t-SNE
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42, n_jobs=-1)
    embeddings = tsne.fit_transform(features)

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=embeddings[labels == digit, 0],
        y=embeddings[labels == digit, 1],
        z=embeddings[labels == digit, 2],
        mode='markers',
        name=f'Digit {digit}',
        marker=dict(
            size=4,
            opacity=0.7
        )
    ) for digit in range(10)])

    fig.update_layout(
        title='3D t-SNE Visualization of Features',
        scene=dict(
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            zaxis_title='t-SNE 3'
        ),
        width=900,
        height=700
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path.with_suffix('.html')))

    return fig


def plot_pca(
    features: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create PCA visualization (faster alternative to t-SNE).

    Args:
        features: Feature array (N, D)
        labels: Labels array (N,)
        n_components: Number of PCA components
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    print(f"Running PCA on {len(features)} samples...")

    # Apply PCA
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(features)

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=labels,
            cmap='tab10',
            s=20,
            alpha=0.7
        )

        plt.colorbar(scatter, label='Digit', ax=ax)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_title('PCA Visualization of Features')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    else:
        # Multiple components
        fig, axes = plt.subplots(1, n_components-1, figsize=(6*(n_components-1), 5))

        for i in range(n_components-1):
            scatter = axes[i].scatter(
                embeddings[:, i],
                embeddings[:, i+1],
                c=labels,
                cmap='tab10',
                s=20,
                alpha=0.7
            )

            axes[i].set_xlabel(f'PC{i+1} ({pca.explained_variance_ratio_[i]:.2%})')
            axes[i].set_ylabel(f'PC{i+2} ({pca.explained_variance_ratio_[i+1]:.2%})')
            axes[i].set_title(f'PC{i+1} vs PC{i+2}')
            axes[i].grid(True, alpha=0.3)

        plt.colorbar(scatter, label='Digit', ax=axes[-1])
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_explained_variance(
    features: np.ndarray,
    max_components: int = 10,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot PCA explained variance.

    Args:
        features: Feature array (N, D)
        max_components: Maximum PCA components to consider
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    max_components = min(max_components, features.shape[1])

    pca = PCA(n_components=max_components)
    pca.fit(features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Individual explained variance
    axes[0].bar(range(1, max_components+1), pca.explained_variance_ratio_)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Individual Explained Variance')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Cumulative explained variance
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, max_components+1), cumulative_var, 'bo-')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95%')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def compare_layers_tsne(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    layers: list = ['classical', 'quantum', 'final'],
    max_samples: int = 500,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Compare t-SNE visualizations from different layers.

    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device
        layers: Layers to compare
        max_samples: Maximum samples
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, len(layers), figsize=(6*len(layers), 5))

    if len(layers) == 1:
        axes = [axes]

    for idx, layer in enumerate(layers):
        print(f"\nProcessing {layer} layer...")

        # Extract features
        features, labels = extract_features(
            model, dataloader, device,
            max_samples=max_samples,
            layer=layer
        )

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
        embeddings = tsne.fit_transform(features)

        # Plot
        scatter = axes[idx].scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=labels,
            cmap='tab10',
            s=20,
            alpha=0.7
        )

        axes[idx].set_xlabel('t-SNE 1')
        axes[idx].set_ylabel('t-SNE 2')
        axes[idx].set_title(f'{layer.title()} Layer Features')
        axes[idx].grid(True, alpha=0.3)

    plt.colorbar(scatter, label='Digit', ax=axes[-1])
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    print("Embeddings visualization module ready!")
    print("✓ t-SNE 2D visualization")
    print("✓ t-SNE 3D visualization")
    print("✓ PCA visualization")
    print("✓ Explained variance analysis")
    print("✓ Multi-layer comparison")
