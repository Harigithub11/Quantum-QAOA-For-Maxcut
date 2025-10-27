"""
MNIST Dataset Loading and Preprocessing
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class MNISTDataModule:
    """
    Data module for MNIST dataset with preprocessing and data loaders.
    """

    def __init__(
        self,
        data_dir: str = "./data/raw",
        batch_size: int = 64,
        validation_split: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42
    ):
        """
        Initialize MNIST data module.

        Args:
            data_dir: Directory to store/load MNIST data
            batch_size: Batch size for data loaders
            validation_split: Fraction of training data to use for validation
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])

        # Placeholders for datasets and loaders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self):
        """
        Download and setup MNIST datasets and create data loaders.
        """
        # Download/load full training dataset
        full_train_dataset = datasets.MNIST(
            root=str(self.data_dir),
            train=True,
            transform=self.transform,
            download=True
        )

        # Download/load test dataset
        self.test_dataset = datasets.MNIST(
            root=str(self.data_dir),
            train=False,
            transform=self.transform,
            download=True
        )

        # Split training data into train and validation
        val_size = int(len(full_train_dataset) * self.validation_split)
        train_size = len(full_train_dataset) - val_size

        # Use generator for reproducibility
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=generator
        )

        print(f"Dataset sizes:")
        print(f"  Training: {len(self.train_dataset)}")
        print(f"  Validation: {len(self.val_dataset)}")
        print(f"  Test: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        """
        Create training data loader.

        Returns:
            DataLoader for training data
        """
        if self.train_dataset is None:
            self.setup()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  # Drop last incomplete batch
        )
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """
        Create validation data loader.

        Returns:
            DataLoader for validation data
        """
        if self.val_dataset is None:
            self.setup()

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        """
        Create test data loader.

        Returns:
            DataLoader for test data
        """
        if self.test_dataset is None:
            self.setup()

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        return self.test_loader

    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample batch for testing/visualization.

        Returns:
            Tuple of (images, labels)
        """
        if self.train_loader is None:
            self.train_dataloader()

        return next(iter(self.train_loader))

    def get_class_names(self) -> list:
        """
        Get MNIST class names (digit labels).

        Returns:
            List of class names
        """
        return [str(i) for i in range(10)]


def get_mnist_dataloaders(
    data_dir: str = "./data/raw",
    batch_size: int = 64,
    validation_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to get all data loaders at once.

    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for data loaders
        validation_split: Fraction of training data to use for validation
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_module = MNISTDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        validation_split=validation_split,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed
    )

    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data module
    print("Testing MNIST Data Module...")

    data_module = MNISTDataModule(batch_size=32)
    data_module.setup()

    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Test a batch
    images, labels = data_module.get_sample_batch()
    print(f"\nSample batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Labels: {labels[:10].tolist()}")

    print("\nData module test successful!")
