"""
Training Utilities
Device management, seed setting, configuration loading, and helper functions
"""

import torch
import numpy as np
import random
import yaml
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed} for reproducibility")


def get_device(use_cuda: bool = True, device_id: int = 0) -> torch.device:
    """
    Get available device (GPU/CPU) for training.

    Args:
        use_cuda: Whether to use CUDA if available
        device_id: GPU device ID (for multi-GPU systems)

    Returns:
        torch.device object
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f"\n{'='*70}")
        print(f"GPU Device: {torch.cuda.get_device_name(device_id)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device ID: {device_id}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(device_id) / 1024**3:.2f} GB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(device_id) / 1024**3:.2f} GB")
        print(f"Total Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.2f} GB")
        print(f"{'='*70}\n")
    else:
        device = torch.device('cpu')
        if use_cuda and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Using CPU instead.")
        else:
            print("Using CPU for training")

    return device


def get_gpu_memory_stats(device: torch.device) -> Dict[str, float]:
    """
    Get current GPU memory statistics.

    Args:
        device: PyTorch device

    Returns:
        Dictionary with memory statistics in GB
    """
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device.index) / 1024**3
        reserved = torch.cuda.memory_reserved(device.index) / 1024**3
        total = torch.cuda.get_device_properties(device.index).total_memory / 1024**3

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - allocated,
            'utilization': (allocated / total) * 100
        }
    else:
        return {}


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded configuration from: {config_path}")
    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]):
    """
    Save configuration dictionary to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Configuration saved to: {save_path}")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen
    }


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def update_lr(optimizer: torch.optim.Optimizer, new_lr: float):
    """
    Update learning rate in optimizer.

    Args:
        optimizer: PyTorch optimizer
        new_lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print(f"Learning rate updated to: {new_lr:.6f}")


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update statistics with new value.

        Args:
            val: New value
            n: Number of samples (for batch averaging)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


class Timer:
    """
    Simple timer for tracking elapsed time.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset timer."""
        self.start_time = time.time()
        self.elapsed = 0

    def tick(self) -> float:
        """
        Get elapsed time since last tick.

        Returns:
            Elapsed time in seconds
        """
        current_time = time.time()
        self.elapsed = current_time - self.start_time
        self.start_time = current_time
        return self.elapsed

    def total_time(self) -> float:
        """
        Get total elapsed time since creation/reset.

        Returns:
            Total elapsed time in seconds
        """
        return time.time() - self.start_time


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        self.is_better = self._get_comparison_function()

    def _get_comparison_function(self):
        """Get comparison function based on mode."""
        if self.mode == 'min':
            return lambda current, best: current < best - self.min_delta
        else:  # mode == 'max'
            return lambda current, best: current > best + self.min_delta

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value
            epoch: Current epoch number

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.is_better(score, self.best_score):
            # Improvement
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"Early stopping: Improvement detected (best: {self.best_score:.6f})")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"Early stopping: No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\nEarly stopping triggered! Best score: {self.best_score:.6f} at epoch {self.best_epoch}")
                return True

        return False


def create_experiment_dir(base_dir: str = "experiments", experiment_name: Optional[str] = None) -> Path:
    """
    Create directory for experiment outputs.

    Args:
        base_dir: Base directory for all experiments
        experiment_name: Name of experiment (None = timestamp)

    Returns:
        Path to experiment directory
    """
    base_path = Path(base_dir)

    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"exp_{timestamp}"

    exp_dir = base_path / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)

    print(f"Experiment directory created: {exp_dir}")
    return exp_dir


def get_gradient_norm(model: torch.nn.Module) -> float:
    """
    Calculate total gradient norm across all model parameters.

    Args:
        model: PyTorch model

    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0) -> float:
    """
    Clip gradients by norm.

    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm

    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


def save_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    save_path: Union[str, Path]
):
    """
    Save complete training state for resuming.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Training metrics
        save_path: Path to save state
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(state, save_path)


def load_training_state(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load training state from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load to

    Returns:
        Dictionary with epoch and metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {})
    }


if __name__ == "__main__":
    # Test utilities
    print("=" * 70)
    print("Testing Training Utilities")
    print("=" * 70)

    # Test device detection
    print("\n1. Device Detection:")
    device = get_device(use_cuda=True)
    print(f"Selected device: {device}")

    if device.type == 'cuda':
        stats = get_gpu_memory_stats(device)
        print("\nGPU Memory Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")

    # Test seed setting
    print("\n2. Seed Setting:")
    set_seed(42)

    # Test time formatting
    print("\n3. Time Formatting:")
    print(f"  3661 seconds = {format_time(3661)}")
    print(f"  125 seconds = {format_time(125)}")
    print(f"  45 seconds = {format_time(45)}")

    # Test AverageMeter
    print("\n4. AverageMeter:")
    meter = AverageMeter("Loss")
    for i in range(5):
        meter.update(1.0 / (i + 1))
        print(f"  Step {i+1}: {meter}")

    # Test Timer
    print("\n5. Timer:")
    timer = Timer()
    time.sleep(0.1)
    print(f"  Elapsed: {timer.tick():.3f}s")

    # Test EarlyStopping
    print("\n6. Early Stopping (mode='min', patience=3):")
    early_stop = EarlyStopping(patience=3, min_delta=0.01, mode='min', verbose=False)
    test_losses = [1.0, 0.9, 0.85, 0.84, 0.83, 0.835, 0.834]
    for epoch, loss in enumerate(test_losses):
        should_stop = early_stop(loss, epoch)
        print(f"  Epoch {epoch}: loss={loss:.3f}, stop={should_stop}, counter={early_stop.counter}")
        if should_stop:
            break

    print("\n" + "=" * 70)
    print("Utilities test completed!")
    print("=" * 70)
