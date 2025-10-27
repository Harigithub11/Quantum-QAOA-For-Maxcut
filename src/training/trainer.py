"""
Training Engine
GPU-accelerated training loop with MLflow tracking and TensorBoard visualization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils import (
    AverageMeter, Timer, EarlyStopping, get_lr,
    get_gpu_memory_stats, clip_gradients
)
from training.metrics import (
    compute_accuracy, MetricsTracker, evaluate_model,
    compute_confusion_matrix, plot_confusion_matrix,
    get_classification_report, compute_per_class_metrics
)
from training.checkpoint import CheckpointManager


class Trainer:
    """
    Training engine for hybrid quantum-classical model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device('cpu'),
        config: Optional[Dict[str, Any]] = None,
        experiment_name: str = "quantum-mnist",
        checkpoint_dir: str = "models/checkpoints",
        log_dir: str = "logs",
        use_amp: bool = False,
        gradient_clip: Optional[float] = None,
        log_interval: int = 10
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            config: Training configuration
            experiment_name: MLflow experiment name
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
            use_amp: Use automatic mixed precision
            gradient_clip: Gradient clipping max norm
            log_interval: Log every N batches
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}

        self.use_amp = use_amp and device.type == 'cuda'
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval

        # Move model to device
        self.model = self.model.to(device)

        # Automatic Mixed Precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=5,
            monitor_metric="val_accuracy",
            mode="max",
            verbose=True
        )

        # Metrics tracker
        self.metrics_tracker = MetricsTracker([
            'train_loss', 'train_acc',
            'val_loss', 'val_acc',
            'learning_rate', 'epoch_time'
        ])

        # TensorBoard writer
        tensorboard_dir = Path(log_dir) / "tensorboard" / experiment_name
        self.writer = SummaryWriter(str(tensorboard_dir))

        # MLflow setup
        mlflow.set_experiment(experiment_name)

        # Early stopping
        early_stop_config = self.config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stop_config.get('patience', 10),
            min_delta=early_stop_config.get('min_delta', 0.001),
            mode='min',
            verbose=True
        ) if early_stop_config.get('enabled', True) else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0

        print(f"\n{'='*70}")
        print(f"Trainer Initialized")
        print(f"{'='*70}")
        print(f"Device: {device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Clipping: {gradient_clip}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"{'='*70}\n")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()

        loss_meter = AverageMeter("Loss")
        acc_meter = AverageMeter("Accuracy")

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            ncols=100,
            file=sys.stdout
        )

        for batch_idx, (images, targets) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass with AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = clip_gradients(self.model, self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip is not None:
                    grad_norm = clip_gradients(self.model, self.gradient_clip)
                self.optimizer.step()

            # Compute accuracy
            with torch.no_grad():
                acc = compute_accuracy(logits, targets)

            # Update meters
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.2f}%'
            })

            # Log to TensorBoard
            if batch_idx % self.log_interval == 0:
                self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/BatchAcc', acc, self.global_step)

                if self.gradient_clip is not None:
                    self.writer.add_scalar('Train/GradientNorm', grad_norm, self.global_step)

            self.global_step += 1

        return loss_meter.avg, acc_meter.avg

    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate model.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()

        loss_meter = AverageMeter("Val Loss")
        acc_meter = AverageMeter("Val Accuracy")

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in tqdm(
                self.val_loader,
                desc="Validation",
                ncols=100,
                leave=False,
                file=sys.stdout
            ):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward pass
                logits = self.model(images)
                loss = self.criterion(logits, targets)

                # Compute accuracy
                acc = compute_accuracy(logits, targets)
                predictions = torch.argmax(logits, dim=1)

                # Update meters
                batch_size = images.size(0)
                loss_meter.update(loss.item(), batch_size)
                acc_meter.update(acc, batch_size)

                # Collect predictions for confusion matrix
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        # Log confusion matrix every 5 epochs
        if epoch % 5 == 0:
            cm = compute_confusion_matrix(all_predictions, all_targets)
            fig = plot_confusion_matrix(cm, normalize=True)
            self.writer.add_figure('Validation/ConfusionMatrix', fig, epoch)

        return loss_meter.avg, acc_meter.avg

    def train(
        self,
        num_epochs: int,
        start_epoch: int = 0,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
            start_epoch: Starting epoch (for resuming)
            resume_from: Path to checkpoint to resume from

        Returns:
            Dictionary with training history
        """
        # Resume from checkpoint if provided
        if resume_from is not None:
            info = self.checkpoint_manager.load_checkpoint(
                resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
                self.device
            )
            start_epoch = info['epoch'] + 1
            print(f"Resumed training from epoch {start_epoch}")

        # Start MLflow run
        with mlflow.start_run(run_name=f"training_{time.strftime('%Y%m%d_%H%M%S')}"):
            # Log configuration
            mlflow.log_params(self.config)

            # Log model info
            if hasattr(self.model, 'get_model_info'):
                model_info = self.model.get_model_info()
                mlflow.log_params({
                    f"model_{k}": v for k, v in model_info.items()
                    if isinstance(v, (int, float, str, bool))
                })

            print(f"\n{'='*70}")
            print(f"Starting Training")
            print(f"{'='*70}")
            print(f"Epochs: {start_epoch} to {num_epochs}")
            print(f"Device: {self.device}")
            print(f"{'='*70}\n")

            # Training timer
            total_timer = Timer()

            # Training loop
            for epoch in range(start_epoch, num_epochs):
                epoch_timer = Timer()
                self.current_epoch = epoch

                # Train one epoch
                train_loss, train_acc = self.train_epoch(epoch)

                # Validate
                val_loss, val_acc = self.validate(epoch)

                # Learning rate
                current_lr = get_lr(self.optimizer)

                # Scheduler step
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Epoch time
                epoch_time = epoch_timer.tick()

                # Update metrics tracker
                self.metrics_tracker.update(
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    learning_rate=current_lr,
                    epoch_time=epoch_time
                )

                # Log to MLflow
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': current_lr
                }, step=epoch)

                # Log to TensorBoard
                self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
                self.writer.add_scalar('Epoch/TrainAcc', train_acc, epoch)
                self.writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
                self.writer.add_scalar('Epoch/ValAcc', val_acc, epoch)
                self.writer.add_scalar('Epoch/LearningRate', current_lr, epoch)

                # Print epoch summary
                print(f"\nEpoch {epoch}/{num_epochs-1} Summary:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
                print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")

                # GPU memory stats
                if self.device.type == 'cuda':
                    mem_stats = get_gpu_memory_stats(self.device)
                    print(f"  GPU Memory: {mem_stats['allocated_gb']:.2f}/{mem_stats['total_gb']:.2f} GB")

                # Check if best model
                is_best = self.checkpoint_manager.update_best(epoch, {'val_accuracy': val_acc})
                if is_best:
                    self.best_val_acc = val_acc
                    print(f"  *** New best model! Val Acc: {val_acc:.2f}% ***")

                # Save checkpoint
                metrics = {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }

                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    metrics,
                    is_best=is_best
                )

                # Early stopping
                if self.early_stopping is not None:
                    if self.early_stopping(val_loss, epoch):
                        print(f"\nEarly stopping triggered at epoch {epoch}")
                        print(f"Best validation loss: {self.early_stopping.best_score:.4f}")
                        break

                print("-" * 70)

            # Training complete
            total_time = total_timer.total_time()
            print(f"\n{'='*70}")
            print(f"Training Complete!")
            print(f"{'='*70}")
            print(f"Total time: {total_time/60:.2f} minutes")
            print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
            print(f"{'='*70}\n")

            # Save final model
            final_metrics = {
                'final_train_loss': self.metrics_tracker.get_latest('train_loss'),
                'final_train_acc': self.metrics_tracker.get_latest('train_acc'),
                'final_val_loss': self.metrics_tracker.get_latest('val_loss'),
                'final_val_acc': self.metrics_tracker.get_latest('val_acc'),
                'best_val_acc': self.best_val_acc,
                'total_epochs': self.current_epoch + 1,
                'total_time_minutes': total_time / 60
            }

            self.checkpoint_manager.save_final_model(self.model, final_metrics)

            # Log final model to MLflow
            mlflow.pytorch.log_model(self.model, "model")

            # Save and log metrics plots
            results_dir = Path("results/figures")
            results_dir.mkdir(parents=True, exist_ok=True)

            metrics_plot_path = results_dir / "training_metrics.png"
            self.metrics_tracker.plot_metrics(save_path=metrics_plot_path)
            mlflow.log_artifact(str(metrics_plot_path))

            # Return training history
            return {
                'metrics': self.metrics_tracker.to_dict(),
                'best_val_acc': self.best_val_acc,
                'total_time': total_time,
                'final_epoch': self.current_epoch
            }

    def test(self, test_loader: DataLoader, num_classes: int = 10) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader
            num_classes: Number of classes

        Returns:
            Dictionary with test metrics
        """
        print(f"\n{'='*70}")
        print("Evaluating on Test Set")
        print(f"{'='*70}\n")

        # Evaluate
        metrics = evaluate_model(
            self.model,
            test_loader,
            self.criterion,
            self.device,
            num_classes=num_classes,
            return_predictions=True
        )

        print(f"Test Results:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.2f}%")

        # Confusion matrix
        cm = compute_confusion_matrix(
            metrics['predictions'],
            metrics['targets'],
            num_classes
        )

        results_dir = Path("results/figures")
        cm_path = results_dir / "confusion_matrix_test.png"
        plot_confusion_matrix(cm, save_path=cm_path, normalize=True)

        # Classification report
        report_path = Path("results") / "classification_report_test.txt"
        class_names = [str(i) for i in range(num_classes)]
        report = get_classification_report(
            metrics['predictions'],
            metrics['targets'],
            class_names=class_names,
            save_path=report_path
        )

        print(f"\nClassification Report:\n{report}")

        # Log to MLflow
        if mlflow.active_run():
            mlflow.log_metrics({
                'test_loss': metrics['loss'],
                'test_accuracy': metrics['accuracy'],
                'test_top3_accuracy': metrics['top3_accuracy']
            })
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(report_path))

        print(f"\n{'='*70}\n")

        return metrics

    def close(self):
        """Close resources."""
        self.writer.close()
        print("Trainer resources closed")


if __name__ == "__main__":
    print("Trainer module - use train.py script to run training")
