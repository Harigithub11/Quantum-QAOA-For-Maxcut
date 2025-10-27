"""
Training and Evaluation Metrics
Comprehensive metrics for model evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    top_k_accuracy_score
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Predicted class indices (batch_size,) or logits (batch_size, num_classes)
        targets: Ground truth class indices (batch_size,)

    Returns:
        Accuracy as percentage (0-100)
    """
    if predictions.dim() == 2:
        # Convert logits to predictions
        predictions = torch.argmax(predictions, dim=1)

    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    accuracy = (correct / total) * 100.0

    return accuracy


def compute_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    """
    Compute top-k accuracy.

    Args:
        logits: Model output logits (batch_size, num_classes)
        targets: Ground truth class indices (batch_size,)
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy as percentage (0-100)
    """
    with torch.no_grad():
        # Get top k predictions
        _, top_k_preds = logits.topk(k, dim=1, largest=True, sorted=True)

        # Check if target is in top k
        targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).any(dim=1).sum().item()

        accuracy = (correct / targets.size(0)) * 100.0

    return accuracy


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 10
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted class indices (batch_size,) or logits (batch_size, num_classes)
        targets: Ground truth class indices (batch_size,)
        num_classes: Number of classes

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    if predictions.dim() == 2:
        predictions = torch.argmax(predictions, dim=1)

    # Convert to numpy
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()

    cm = confusion_matrix(target_np, pred_np, labels=list(range(num_classes)))
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    normalize: bool = False,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize by row (true labels)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'}
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    return fig


def compute_per_class_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 10,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, F1-score, and accuracy.

    Args:
        predictions: Predicted class indices or logits
        targets: Ground truth class indices
        num_classes: Number of classes
        class_names: List of class names

    Returns:
        Dictionary with metrics for each class
    """
    if predictions.dim() == 2:
        predictions = torch.argmax(predictions, dim=1)

    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    # Compute precision, recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        target_np,
        pred_np,
        labels=list(range(num_classes)),
        zero_division=0
    )

    # Compute per-class accuracy
    per_class_acc = []
    for cls in range(num_classes):
        mask = target_np == cls
        if mask.sum() > 0:
            acc = (pred_np[mask] == cls).sum() / mask.sum()
        else:
            acc = 0.0
        per_class_acc.append(acc * 100)

    # Organize results
    metrics = {}
    for i, class_name in enumerate(class_names):
        metrics[class_name] = {
            'precision': precision[i] * 100,
            'recall': recall[i] * 100,
            'f1_score': f1[i] * 100,
            'accuracy': per_class_acc[i],
            'support': int(support[i])
        }

    return metrics


def get_classification_report(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> str:
    """
    Generate detailed classification report.

    Args:
        predictions: Predicted class indices or logits
        targets: Ground truth class indices
        class_names: List of class names
        save_path: Path to save report

    Returns:
        Classification report as string
    """
    if predictions.dim() == 2:
        predictions = torch.argmax(predictions, dim=1)

    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()

    if class_names is None:
        class_names = None  # sklearn will use numeric labels

    report = classification_report(
        target_np,
        pred_np,
        target_names=class_names,
        digits=4
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to: {save_path}")

    return report


class MetricsTracker:
    """
    Track metrics across training epochs.
    """

    def __init__(self, metrics_to_track: List[str] = None):
        """
        Args:
            metrics_to_track: List of metric names to track
        """
        if metrics_to_track is None:
            metrics_to_track = ['loss', 'accuracy', 'lr']

        self.metrics = {name: [] for name in metrics_to_track}
        self.epoch_count = 0

    def update(self, **kwargs):
        """
        Update metrics for current epoch.

        Args:
            **kwargs: Metric name-value pairs
        """
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

        self.epoch_count += 1

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
            return self.metrics[metric_name][-1]
        return None

    def get_best(self, metric_name: str, mode: str = 'min') -> Tuple[float, int]:
        """
        Get best value and epoch for a metric.

        Args:
            metric_name: Name of metric
            mode: 'min' or 'max'

        Returns:
            Tuple of (best_value, best_epoch)
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return None, -1

        values = self.metrics[metric_name]
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        return values[best_idx], best_idx

    def get_history(self, metric_name: str) -> List[float]:
        """Get full history of a metric."""
        return self.metrics.get(metric_name, [])

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary."""
        return self.metrics.copy()

    def plot_metrics(
        self,
        metrics_to_plot: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot training metrics over epochs.

        Args:
            metrics_to_plot: List of metrics to plot (None = all)
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if metrics_to_plot is None:
            metrics_to_plot = list(self.metrics.keys())

        # Filter out empty metrics
        metrics_to_plot = [m for m in metrics_to_plot if len(self.metrics.get(m, [])) > 0]

        num_metrics = len(metrics_to_plot)
        if num_metrics == 0:
            print("No metrics to plot")
            return None

        # Create subplots
        fig, axes = plt.subplots(num_metrics, 1, figsize=figsize)
        if num_metrics == 1:
            axes = [axes]

        epochs = list(range(1, self.epoch_count + 1))

        for idx, metric_name in enumerate(metrics_to_plot):
            values = self.metrics[metric_name]

            axes[idx].plot(epochs[:len(values)], values, marker='o', linewidth=2)
            axes[idx].set_xlabel('Epoch', fontsize=10)
            axes[idx].set_ylabel(metric_name.replace('_', ' ').title(), fontsize=10)
            axes[idx].set_title(f'{metric_name.replace("_", " ").title()} vs Epoch', fontsize=12)
            axes[idx].grid(True, alpha=0.3)

            # Mark best value
            if 'loss' in metric_name.lower():
                best_val, best_epoch = self.get_best(metric_name, mode='min')
            else:
                best_val, best_epoch = self.get_best(metric_name, mode='max')

            if best_epoch >= 0:
                axes[idx].axvline(
                    x=best_epoch + 1,
                    color='r',
                    linestyle='--',
                    alpha=0.5,
                    label=f'Best: {best_val:.4f}'
                )
                axes[idx].legend()

        plt.tight_layout()

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to: {save_path}")

        return fig


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 10,
    return_predictions: bool = False
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.

    Args:
        model: PyTorch model
        dataloader: Data loader for evaluation
        criterion: Loss function
        device: Device to run evaluation on
        num_classes: Number of classes
        return_predictions: Whether to return predictions and targets

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_logits = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, targets)

            # Collect predictions
            predictions = torch.argmax(logits, dim=1)

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_logits.append(logits.cpu())

            total_loss += loss.item()
            num_batches += 1

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_logits = torch.cat(all_logits, dim=0)

    # Compute metrics
    avg_loss = total_loss / num_batches
    accuracy = compute_accuracy(all_predictions, all_targets)
    top3_accuracy = compute_top_k_accuracy(all_logits, all_targets, k=3)

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'num_samples': len(all_targets)
    }

    if return_predictions:
        metrics['predictions'] = all_predictions
        metrics['targets'] = all_targets
        metrics['logits'] = all_logits

    return metrics


if __name__ == "__main__":
    # Test metrics
    print("=" * 70)
    print("Testing Metrics Module")
    print("=" * 70)

    # Create dummy data
    num_samples = 100
    num_classes = 10

    logits = torch.randn(num_samples, num_classes)
    predictions = torch.argmax(logits, dim=1)
    targets = torch.randint(0, num_classes, (num_samples,))

    # Test accuracy
    print("\n1. Accuracy:")
    acc = compute_accuracy(predictions, targets)
    print(f"  Overall accuracy: {acc:.2f}%")

    acc_from_logits = compute_accuracy(logits, targets)
    print(f"  Accuracy from logits: {acc_from_logits:.2f}%")

    # Test top-k accuracy
    print("\n2. Top-K Accuracy:")
    top3_acc = compute_top_k_accuracy(logits, targets, k=3)
    print(f"  Top-3 accuracy: {top3_acc:.2f}%")

    # Test confusion matrix
    print("\n3. Confusion Matrix:")
    cm = compute_confusion_matrix(predictions, targets, num_classes)
    print(f"  Shape: {cm.shape}")
    print(f"  Sum: {cm.sum()}")

    # Test per-class metrics
    print("\n4. Per-Class Metrics:")
    per_class = compute_per_class_metrics(predictions, targets, num_classes)
    print(f"  Number of classes: {len(per_class)}")
    print(f"  Sample (Class_0):")
    for key, value in per_class['Class_0'].items():
        print(f"    {key}: {value:.2f}")

    # Test classification report
    print("\n5. Classification Report:")
    report = get_classification_report(predictions, targets)
    print(report[:300] + "...")

    # Test metrics tracker
    print("\n6. Metrics Tracker:")
    tracker = MetricsTracker(['train_loss', 'val_loss', 'accuracy'])

    for epoch in range(5):
        tracker.update(
            train_loss=1.0 / (epoch + 1),
            val_loss=1.2 / (epoch + 1),
            accuracy=50 + epoch * 5
        )

    print(f"  Tracked {tracker.epoch_count} epochs")
    print(f"  Latest accuracy: {tracker.get_latest('accuracy'):.2f}")

    best_val, best_epoch = tracker.get_best('val_loss', mode='min')
    print(f"  Best val_loss: {best_val:.4f} at epoch {best_epoch + 1}")

    print("\n" + "=" * 70)
    print("Metrics test completed!")
    print("=" * 70)
