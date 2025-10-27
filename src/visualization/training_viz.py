"""
Advanced Training Visualizations
Interactive Plotly dashboards and matplotlib plots for training analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    interactive: bool = True
) -> go.Figure:
    """
    Create interactive training curves with Plotly.

    Args:
        history: Training history dict with metrics
        save_path: Path to save figure
        interactive: Use Plotly (True) or Matplotlib (False)

    Returns:
        Plotly figure
    """
    epochs = list(range(1, len(history.get('train_loss', [])) + 1))

    if interactive:
        # Create Plotly figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training & Validation Loss', 'Training & Validation Accuracy',
                          'Learning Rate Schedule', 'Epoch Time'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )

        # Loss plot
        if 'train_loss' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss',
                          mode='lines+markers', line=dict(color='blue', width=2)),
                row=1, col=1
            )
        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss',
                          mode='lines+markers', line=dict(color='red', width=2)),
                row=1, col=1
            )

        # Accuracy plot
        if 'train_acc' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc',
                          mode='lines+markers', line=dict(color='blue', width=2)),
                row=1, col=2
            )
        if 'val_acc' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc',
                          mode='lines+markers', line=dict(color='red', width=2)),
                row=1, col=2
            )

        # Learning rate
        if 'learning_rate' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['learning_rate'], name='LR',
                          mode='lines+markers', line=dict(color='green', width=2)),
                row=2, col=1
            )

        # Epoch time
        if 'epoch_time' in history:
            fig.add_trace(
                go.Bar(x=epochs, y=history['epoch_time'], name='Time (s)',
                      marker_color='orange'),
                row=2, col=2
            )

        # Update layout
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)

        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
        fig.update_yaxes(title_text="Learning Rate", row=2, col=1, type="log")
        fig.update_yaxes(title_text="Time (seconds)", row=2, col=2)

        fig.update_layout(
            title_text="Training Metrics Dashboard",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path.with_suffix('.html')))
            fig.write_image(str(save_path), width=1200, height=800)

        return fig

    else:
        # Matplotlib version
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        if 'train_loss' in history:
            axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
        if 'val_loss' in history:
            axes[0, 0].plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        if 'train_acc' in history:
            axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc')
        if 'val_acc' in history:
            axes[0, 1].plot(epochs, history['val_acc'], 'r-o', label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training & Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate
        if 'learning_rate' in history:
            axes[1, 0].plot(epochs, history['learning_rate'], 'g-o')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)

        # Epoch time
        if 'epoch_time' in history:
            axes[1, 1].bar(epochs, history['epoch_time'], color='orange', alpha=0.7)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].set_title('Time per Epoch')
            axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


def plot_metric_comparison(
    metrics_dict: Dict[str, Dict[str, List[float]]],
    metric_name: str = 'val_acc',
    save_path: Optional[Path] = None
) -> go.Figure:
    """
    Compare a specific metric across multiple experiments.

    Args:
        metrics_dict: Dict mapping experiment names to their history
        metric_name: Metric to compare
        save_path: Path to save figure

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for exp_name, history in metrics_dict.items():
        if metric_name in history:
            epochs = list(range(1, len(history[metric_name]) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history[metric_name],
                    name=exp_name,
                    mode='lines+markers'
                )
            )

    fig.update_layout(
        title=f'{metric_name.replace("_", " ").title()} Comparison',
        xaxis_title='Epoch',
        yaxis_title=metric_name.replace('_', ' ').title(),
        hovermode='x unified',
        height=600
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path.with_suffix('.html')))
        fig.write_image(str(save_path))

    return fig


def create_training_summary(
    history: Dict[str, List[float]],
    test_metrics: Dict[str, float],
    save_path: Optional[Path] = None
) -> Dict:
    """
    Create comprehensive training summary.

    Args:
        history: Training history
        test_metrics: Final test metrics
        save_path: Path to save summary

    Returns:
        Summary dictionary
    """
    summary = {
        'training': {
            'total_epochs': len(history.get('train_loss', [])),
            'final_train_loss': history.get('train_loss', [None])[-1],
            'final_train_acc': history.get('train_acc', [None])[-1],
            'final_val_loss': history.get('val_loss', [None])[-1],
            'final_val_acc': history.get('val_acc', [None])[-1],
            'best_val_acc': max(history.get('val_acc', [0])) if history.get('val_acc') else None,
            'best_val_loss': min(history.get('val_loss', [float('inf')])) if history.get('val_loss') else None,
        },
        'test': test_metrics,
        'convergence': {
            'epochs_to_best': (np.argmax(history.get('val_acc', [0])) + 1) if history.get('val_acc') else None,
            'train_val_gap': (history.get('train_acc', [None])[-1] - history.get('val_acc', [None])[-1]) if history.get('train_acc') and history.get('val_acc') else None
        }
    }

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    # Test with dummy data
    print("Testing training visualization module...")

    dummy_history = {
        'train_loss': [2.3, 2.1, 1.8, 1.5, 1.2],
        'val_loss': [2.4, 2.2, 1.9, 1.7, 1.5],
        'train_acc': [20, 35, 50, 65, 75],
        'val_acc': [18, 32, 48, 62, 72],
        'learning_rate': [0.001, 0.001, 0.001, 0.0001, 0.0001],
        'epoch_time': [120, 115, 118, 116, 117]
    }

    fig = plot_training_curves(dummy_history, save_path=Path('test_training.png'))
    print("✓ Training curves created")

    test_metrics = {'test_loss': 1.6, 'test_accuracy': 70.5}
    summary = create_training_summary(dummy_history, test_metrics)
    print(f"✓ Training summary: {summary}")

    print("\nVisualization module test complete!")
