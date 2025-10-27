"""
Model Checkpoint Management
Save, load, and manage model checkpoints during training
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any, List
import shutil
import json
from datetime import datetime


class CheckpointManager:
    """
    Manage model checkpoints during training.
    Handles saving, loading, and tracking best models.
    """

    def __init__(
        self,
        checkpoint_dir: str = "models/checkpoints",
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        monitor_metric: str = "val_accuracy",
        mode: str = "max",
        verbose: bool = True
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep (None = keep all)
            save_best_only: Only save when metric improves
            monitor_metric: Metric to monitor for best model
            mode: 'min' or 'max' for metric comparison
            verbose: Print messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.verbose = verbose

        self.checkpoints = []  # List of checkpoint paths
        self.best_metric = None
        self.best_epoch = -1

        # Comparison function
        if mode == 'min':
            self.is_better = lambda current, best: current < best
        else:
            self.is_better = lambda current, best: current > best

        if self.verbose:
            print(f"CheckpointManager initialized:")
            print(f"  Directory: {self.checkpoint_dir}")
            print(f"  Monitor: {monitor_metric} ({mode})")
            print(f"  Save best only: {save_best_only}")
            print(f"  Max checkpoints: {max_checkpoints}")

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save model checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best model
            filename: Custom filename (None = auto-generate)

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Add model configuration if available
        if hasattr(model, 'get_model_info'):
            checkpoint['model_info'] = model.get_model_info()

        # Generate filename
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pth"

        save_path = self.checkpoint_dir / filename

        # Save checkpoint
        torch.save(checkpoint, save_path)

        if self.verbose:
            print(f"Checkpoint saved: {save_path}")

        # Track checkpoint
        if not is_best:  # Best model has separate tracking
            self.checkpoints.append(save_path)
            self._cleanup_old_checkpoints()

        # Save as best model if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            shutil.copy(save_path, best_path)
            if self.verbose:
                print(f"Best model updated: {best_path}")

        return save_path

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if exceeding max_checkpoints."""
        if self.max_checkpoints is None:
            return

        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                if self.verbose:
                    print(f"Removed old checkpoint: {old_checkpoint}")

    def should_save(self, metrics: Dict[str, float]) -> bool:
        """
        Check if checkpoint should be saved based on metric.

        Args:
            metrics: Current metrics dictionary

        Returns:
            True if should save
        """
        if not self.save_best_only:
            return True

        if self.monitor_metric not in metrics:
            if self.verbose:
                print(f"Warning: Monitor metric '{self.monitor_metric}' not found in metrics")
            return False

        current_metric = metrics[self.monitor_metric]

        if self.best_metric is None:
            return True

        return self.is_better(current_metric, self.best_metric)

    def update_best(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Update best metric tracking.

        Args:
            epoch: Current epoch
            metrics: Current metrics

        Returns:
            True if this is a new best
        """
        if self.monitor_metric not in metrics:
            return False

        current_metric = metrics[self.monitor_metric]

        if self.best_metric is None or self.is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.best_epoch = epoch
            return True

        return False

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to map tensors to

        Returns:
            Dictionary with checkpoint information
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Restore model
        model.load_state_dict(checkpoint['model_state_dict'])

        # Restore optimizer
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore scheduler
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.verbose:
            print(f"Checkpoint loaded: {checkpoint_path}")
            print(f"  Epoch: {checkpoint['epoch']}")
            if 'metrics' in checkpoint:
                print(f"  Metrics: {checkpoint['metrics']}")

        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint.get('metrics', {}),
            'model_info': checkpoint.get('model_info', {}),
            'timestamp': checkpoint.get('timestamp', 'unknown')
        }

    def load_best_model(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load best model checkpoint.

        Args:
            model: Model to load state into
            device: Device to map tensors to

        Returns:
            Dictionary with checkpoint information
        """
        best_path = self.checkpoint_dir / "best_model.pth"

        if not best_path.exists():
            raise FileNotFoundError("Best model checkpoint not found")

        return self.load_checkpoint(best_path, model, device=device)

    def get_best_model_path(self) -> Optional[Path]:
        """Get path to best model checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pth"
        return best_path if best_path.exists() else None

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint."""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]

    def list_checkpoints(self) -> List[Path]:
        """List all checkpoint files."""
        return sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))

    def save_final_model(
        self,
        model: nn.Module,
        metrics: Dict[str, float],
        filename: str = "final_model.pth"
    ):
        """
        Save final model after training completes.

        Args:
            model: Trained model
            metrics: Final metrics
            filename: Filename for final model
        """
        save_path = self.checkpoint_dir / filename

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        if hasattr(model, 'get_model_info'):
            checkpoint['model_info'] = model.get_model_info()

        torch.save(checkpoint, save_path)

        if self.verbose:
            print(f"Final model saved: {save_path}")

    def export_model_for_inference(
        self,
        model: nn.Module,
        save_path: str,
        example_input: Optional[torch.Tensor] = None
    ):
        """
        Export model for inference (model weights only).

        Args:
            model: Model to export
            save_path: Path to save exported model
            example_input: Example input for tracing (optional)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state dict only
        torch.save(model.state_dict(), save_path)

        if self.verbose:
            print(f"Model exported for inference: {save_path}")

        # Optionally save as TorchScript
        if example_input is not None:
            scripted_path = save_path.with_suffix('.pt')
            model.eval()
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save(str(scripted_path))
            if self.verbose:
                print(f"TorchScript model saved: {scripted_path}")

    def save_training_summary(self, summary: Dict[str, Any], filename: str = "training_summary.json"):
        """
        Save training summary to JSON.

        Args:
            summary: Dictionary with training summary
            filename: Filename for summary
        """
        save_path = self.checkpoint_dir / filename

        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)

        if self.verbose:
            print(f"Training summary saved: {save_path}")

    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get information about a checkpoint without loading model.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'model_info': checkpoint.get('model_info', {})
        }

        return info


def resume_training(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Convenience function to resume training from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Scheduler instance (optional)
        device: Device to load to

    Returns:
        Dictionary with epoch and metrics to resume from
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Training resumed from epoch {checkpoint['epoch']}")

    return {
        'start_epoch': checkpoint['epoch'] + 1,
        'metrics': checkpoint.get('metrics', {})
    }


if __name__ == "__main__":
    # Test checkpoint manager
    print("=" * 70)
    print("Testing Checkpoint Manager")
    print("=" * 70)

    # Create dummy model and optimizer
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

    # Initialize checkpoint manager
    print("\n1. Initialize Checkpoint Manager:")
    checkpoint_dir = "test_checkpoints"
    manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=3,
        monitor_metric="val_accuracy",
        mode="max"
    )

    # Simulate training and save checkpoints
    print("\n2. Saving Checkpoints:")
    for epoch in range(5):
        metrics = {
            'train_loss': 1.0 / (epoch + 1),
            'val_loss': 1.2 / (epoch + 1),
            'val_accuracy': 50 + epoch * 10
        }

        is_best = manager.update_best(epoch, metrics)
        should_save = manager.should_save(metrics)

        if should_save:
            manager.save_checkpoint(
                model, optimizer, scheduler,
                epoch=epoch,
                metrics=metrics,
                is_best=is_best
            )

    # List checkpoints
    print("\n3. List Checkpoints:")
    checkpoints = manager.list_checkpoints()
    print(f"  Found {len(checkpoints)} checkpoints")
    for cp in checkpoints:
        print(f"    - {cp.name}")

    # Check best model
    print("\n4. Best Model:")
    best_path = manager.get_best_model_path()
    print(f"  Best model path: {best_path}")
    print(f"  Best metric: {manager.best_metric}")
    print(f"  Best epoch: {manager.best_epoch}")

    # Load checkpoint
    print("\n5. Load Checkpoint:")
    if checkpoints:
        info = manager.get_checkpoint_info(str(checkpoints[0]))
        print(f"  Checkpoint info: {info}")

    # Save final model
    print("\n6. Save Final Model:")
    final_metrics = {'test_accuracy': 95.5, 'test_loss': 0.15}
    manager.save_final_model(model, final_metrics)

    # Cleanup test checkpoints
    print("\n7. Cleanup:")
    import shutil
    if Path(checkpoint_dir).exists():
        shutil.rmtree(checkpoint_dir)
        print(f"  Test checkpoint directory removed: {checkpoint_dir}")

    print("\n" + "=" * 70)
    print("Checkpoint manager test completed!")
    print("=" * 70)
