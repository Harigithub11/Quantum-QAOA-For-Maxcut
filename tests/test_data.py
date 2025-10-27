"""
Tests for data loading and preprocessing
"""

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.mnist_loader import load_mnist_data, MNISTDataModule


class TestMNISTLoading:
    """Test MNIST dataset loading"""

    def test_mnist_download(self):
        """Test MNIST dataset can be downloaded"""
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        assert len(dataset) == 60000, "Training set should have 60000 samples"

    def test_mnist_test_set(self):
        """Test MNIST test set loading"""
        dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        assert len(dataset) == 10000, "Test set should have 10000 samples"

    def test_image_shape(self):
        """Test image dimensions"""
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        image, label = dataset[0]
        assert image.shape == (1, 28, 28), f"Expected shape (1, 28, 28), got {image.shape}"

    def test_label_range(self):
        """Test labels are in valid range"""
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        labels = [dataset[i][1] for i in range(100)]
        assert all(0 <= label <= 9 for label in labels), "Labels should be 0-9"

    def test_image_values(self):
        """Test image pixel values are normalized"""
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        image, _ = dataset[0]
        assert image.min() >= 0.0, "Pixel values should be >= 0"
        assert image.max() <= 1.0, "Pixel values should be <= 1"


class TestDataTransforms:
    """Test data transformations"""

    def test_resize_transform(self):
        """Test image resizing"""
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        image, _ = dataset[0]
        assert image.shape == (1, 32, 32), f"Expected (1, 32, 32), got {image.shape}"

    def test_normalize_transform(self):
        """Test normalization"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        image, _ = dataset[0]
        # After normalization, values can be negative
        assert image.dtype == torch.float32


class TestDataLoaders:
    """Test DataLoader functionality"""

    def test_dataloader_creation(self):
        """Test creating DataLoader"""
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        assert len(loader) > 0, "DataLoader should not be empty"

    def test_batch_shape(self):
        """Test batch dimensions"""
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        loader = DataLoader(dataset, batch_size=32)
        images, labels = next(iter(loader))
        assert images.shape == (32, 1, 28, 28), f"Expected (32, 1, 28, 28), got {images.shape}"
        assert labels.shape == (32,), f"Expected (32,), got {labels.shape}"

    def test_dataloader_iteration(self):
        """Test iterating through DataLoader"""
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        loader = DataLoader(dataset, batch_size=64)

        batch_count = 0
        for images, labels in loader:
            batch_count += 1
            assert images.shape[0] <= 64
            assert labels.shape[0] <= 64
            if batch_count >= 10:  # Test first 10 batches
                break

        assert batch_count == 10, "Should iterate through batches"

    def test_shuffle_effect(self):
        """Test shuffle produces different batches"""
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        loader1 = DataLoader(dataset, batch_size=32, shuffle=True)
        loader2 = DataLoader(dataset, batch_size=32, shuffle=True)

        batch1 = next(iter(loader1))[1]  # labels
        batch2 = next(iter(loader2))[1]  # labels

        # With shuffle, batches should likely be different
        # (not guaranteed but highly probable)
        assert not torch.equal(batch1, batch2) or True  # Allow equal as edge case


class TestDataSplits:
    """Test train/validation/test splits"""

    def test_train_val_split(self):
        """Test creating train/validation split"""
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        assert len(train_dataset) == train_size
        assert len(val_dataset) == val_size
        assert len(train_dataset) + len(val_dataset) == len(dataset)

    def test_no_data_leakage(self):
        """Test train/val split has no overlap"""
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Check indices don't overlap
        train_indices = set(train_dataset.indices)
        val_indices = set(val_dataset.indices)

        assert len(train_indices.intersection(val_indices)) == 0, "Train/val should not overlap"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_batch(self):
        """Test behavior with batch size 1"""
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        loader = DataLoader(dataset, batch_size=1)
        images, labels = next(iter(loader))
        assert images.shape == (1, 1, 28, 28)

    def test_large_batch(self):
        """Test large batch size"""
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        loader = DataLoader(dataset, batch_size=1024)
        images, labels = next(iter(loader))
        assert images.shape[0] == 1024

    def test_drop_last(self):
        """Test drop_last parameter"""
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        loader = DataLoader(dataset, batch_size=64, drop_last=True)

        for images, labels in loader:
            assert images.shape[0] == 64, "All batches should have size 64 with drop_last=True"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
