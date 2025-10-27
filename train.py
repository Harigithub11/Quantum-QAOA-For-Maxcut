"""
Main Training Script
Command-line interface for training the hybrid quantum-classical model
"""

import torch
import torch.nn as nn
import argparse
import yaml
from pathlib import Path
import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.dataset import MNISTDataModule
from src.models.hybrid import HybridQuantumClassifier
from src.training.trainer import Trainer
from src.training.utils import set_seed, get_device, load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Hybrid Quantum-Classical Model for MNIST Classification"
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for training (overrides config)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )

    # Model parameters
    parser.add_argument(
        '--quantum-layers',
        type=int,
        default=None,
        help='Number of quantum variational layers (overrides config)'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=None,
        help='Use pretrained ResNet18'
    )
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Do not use pretrained ResNet18'
    )

    # Device parameters
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA (use CPU only)'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU device ID'
    )

    # Training options
    parser.add_argument(
        '--use-amp',
        action='store_true',
        help='Use automatic mixed precision training'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    # Configuration files
    parser.add_argument(
        '--model-config',
        type=str,
        default='configs/model_config.yaml',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--training-config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training configuration file'
    )

    # Experiment tracking
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='quantum-mnist-hybrid',
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='MLflow run name'
    )

    # Evaluation
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only run evaluation on test set'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint to use for testing'
    )

    return parser.parse_args()


def load_configs(args):
    """Load and merge configurations."""
    # Load YAML configs
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)

    # Merge configs
    config = {
        'model': model_config,
        'training': training_config
    }

    # Override with command-line arguments
    if args.epochs is not None:
        config['training']['training']['num_epochs'] = args.epochs

    if args.batch_size is not None:
        config['training']['data']['batch_size'] = args.batch_size

    if args.learning_rate is not None:
        config['training']['training']['learning_rate'] = args.learning_rate

    if args.quantum_layers is not None:
        config['model']['quantum']['n_layers'] = args.quantum_layers

    if args.no_pretrained:
        config['model']['classical']['pretrained'] = False
    elif args.pretrained:
        config['model']['classical']['pretrained'] = True

    return config


def create_model(config: dict, device: torch.device) -> HybridQuantumClassifier:
    """Create hybrid model from configuration."""
    model_config = config['model']

    model = HybridQuantumClassifier(
        pretrained_resnet=model_config['classical']['pretrained'],
        freeze_resnet=model_config['classical']['freeze_layers'],
        feature_dim=model_config['classical']['feature_dim'],
        dropout=model_config['classical'].get('dropout', 0.2),
        n_qubits=model_config['quantum']['n_qubits'],
        n_layers=model_config['quantum']['n_layers'],
        quantum_device=model_config['quantum']['device'],
        diff_method=model_config['quantum']['diff_method'],
        n_classes=model_config['output']['n_classes'],
        hidden_dim=16
    )

    return model


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer from configuration."""
    training_config = config['training']['training']

    optimizer_config = training_config.get('optimizer', {})
    optimizer_name = optimizer_config.get('name', 'adam').lower()

    learning_rate = training_config['learning_rate']
    weight_decay = training_config.get('weight_decay', 0.0)

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-8)
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Create learning rate scheduler from configuration."""
    training_config = config['training']['training']
    scheduler_config = training_config.get('scheduler', {})

    scheduler_name = scheduler_config.get('name', 'step_lr').lower()

    if scheduler_name == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5),
            verbose=True
        )
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config['num_epochs'],
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    else:
        scheduler = None

    return scheduler


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load configurations
    print("\n" + "="*70)
    print("Hybrid Quantum-Classical MNIST Training")
    print("="*70)

    config = load_configs(args)

    # Get device
    use_cuda = not args.no_cuda
    device = get_device(use_cuda=use_cuda, device_id=args.gpu_id)

    # Create data module
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)

    data_config = config['training']['data']
    data_module = MNISTDataModule(
        data_dir='data/raw',
        batch_size=data_config['batch_size'],
        num_workers=data_config.get('num_workers', 4),
        validation_split=data_config.get('validation_split', 0.1),
        pin_memory=data_config.get('pin_memory', True) and use_cuda
    )

    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    print("\n" + "="*70)
    print("Creating Model")
    print("="*70)

    model = create_model(config, device)
    model = model.to(device)

    # Print model info
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print(f"\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    # Test only mode
    if args.test_only:
        print("\n" + "="*70)
        print("Test Only Mode")
        print("="*70)

        checkpoint_path = args.checkpoint or "models/checkpoints/best_model.pth"

        if not Path(checkpoint_path).exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            return

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {checkpoint_path}")

        # Create trainer for testing
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            experiment_name=args.experiment_name
        )

        # Run test
        test_metrics = trainer.test(test_loader, num_classes=10)

        print(f"\nTest complete!")
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")

        trainer.close()
        return

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer = create_optimizer(model, config)
    print(f"\nOptimizer: {optimizer.__class__.__name__}")
    print(f"Learning rate: {config['training']['training']['learning_rate']}")

    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    if scheduler is not None:
        print(f"Scheduler: {scheduler.__class__.__name__}")

    # Create trainer
    print("\n" + "="*70)
    print("Initializing Trainer")
    print("="*70)

    gradient_clip_config = config['training']['training'].get('gradient_clipping', {})
    gradient_clip = gradient_clip_config.get('max_norm') if gradient_clip_config.get('enabled', True) else None

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        experiment_name=args.experiment_name,
        checkpoint_dir='models/checkpoints',
        log_dir='logs',
        use_amp=args.use_amp,
        gradient_clip=gradient_clip,
        log_interval=config['training']['logging'].get('log_interval', 10)
    )

    # Train model
    num_epochs = config['training']['training']['num_epochs']

    try:
        history = trainer.train(
            num_epochs=num_epochs,
            resume_from=args.resume
        )

        # Save training history
        history_path = Path("results") / "training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\nTraining history saved to: {history_path}")

        # Test on test set
        print("\n" + "="*70)
        print("Testing Best Model")
        print("="*70)

        # Load best model
        best_model_path = "models/checkpoints/best_model.pth"
        if Path(best_model_path).exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from: {best_model_path}")

        # Run test
        test_metrics = trainer.test(test_loader, num_classes=10)

        # Save test results
        test_results_path = Path("results") / "test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump({
                'test_loss': test_metrics['loss'],
                'test_accuracy': test_metrics['accuracy'],
                'test_top3_accuracy': test_metrics['top3_accuracy'],
                'num_samples': test_metrics['num_samples']
            }, f, indent=2)

        print(f"\nTest results saved to: {test_results_path}")

        # Final summary
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Best Validation Accuracy: {history['best_val_acc']:.2f}%")
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"Total Training Time: {history['total_time']/60:.2f} minutes")
        print(f"Checkpoints saved in: models/checkpoints/")
        print(f"Results saved in: results/")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Partial results and checkpoints have been saved.")

    finally:
        # Close trainer resources
        trainer.close()


if __name__ == "__main__":
    main()
