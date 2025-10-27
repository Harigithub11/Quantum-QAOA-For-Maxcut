"""
Tests for training infrastructure
"""

import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.utils import set_seed, get_device, AverageMeter, Timer, EarlyStopping
from src.training.metrics import compute_accuracy, MetricsTracker


class TestTrainingUtils:
    """Test training utility functions"""

    def test_set_seed(self):
        """Test seed setting for reproducibility"""
        set_seed(42)
        val1 = torch.randn(5)

        set_seed(42)
        val2 = torch.randn(5)

        assert torch.equal(val1, val2), "Same seed should produce same random values"

    def test_get_device_cpu(self):
        """Test device selection (CPU)"""
        device = get_device(use_cuda=False)
        assert device.type == 'cpu'

    def test_average_meter(self):
        """Test AverageMeter"""
        meter = AverageMeter()

        meter.update(10)
        assert meter.avg == 10

        meter.update(20)
        assert meter.avg == 15  # (10 + 20) / 2

        meter.update(30, n=2)
        assert meter.count == 4

    def test_timer(self):
        """Test Timer"""
        import time

        timer = Timer()
        time.sleep(0.1)
        elapsed = timer.elapsed()

        assert elapsed >= 0.1, "Timer should measure time correctly"

    def test_early_stopping(self):
        """Test EarlyStopping"""
        early_stop = EarlyStopping(patience=3, mode='max')

        # Improving scores
        assert not early_stop(0.8)
        assert not early_stop(0.85)
        assert not early_stop(0.9)

        # No improvement
        assert not early_stop(0.89)
        assert not early_stop(0.88)
        assert early_stop(0.87)  # Should trigger after 3 epochs without improvement


class TestMetrics:
    """Test metrics computation"""

    def test_compute_accuracy(self):
        """Test accuracy computation"""
        outputs = torch.tensor([[2.0, 1.0, 0.1],
                               [0.1, 2.0, 1.0],
                               [1.0, 0.1, 2.0]])
        targets = torch.tensor([0, 1, 2])

        acc = compute_accuracy(outputs, targets)
        assert acc == 100.0, "All predictions correct should give 100%"

    def test_compute_accuracy_wrong(self):
        """Test accuracy with wrong predictions"""
        outputs = torch.tensor([[1.0, 2.0],
                               [2.0, 1.0]])
        targets = torch.tensor([0, 0])

        acc = compute_accuracy(outputs, targets)
        assert acc == 50.0, "50% correct should give 50%"

    def test_metrics_tracker(self):
        """Test MetricsTracker"""
        tracker = MetricsTracker()

        tracker.update('train_loss', 2.0)
        tracker.update('train_loss', 1.8)
        tracker.update('train_loss', 1.6)

        history = tracker.get_history()
        assert 'train_loss' in history
        assert len(history['train_loss']) == 3
        assert history['train_loss'][-1] == 1.6


class TestLoss:
    """Test loss functions"""

    def test_cross_entropy_loss(self):
        """Test CrossEntropyLoss"""
        criterion = torch.nn.CrossEntropyLoss()

        outputs = torch.tensor([[2.0, 1.0, 0.1]])
        targets = torch.tensor([0])

        loss = criterion(outputs, targets)
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_loss_gradients(self):
        """Test loss backpropagation"""
        criterion = torch.nn.CrossEntropyLoss()

        outputs = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        targets = torch.tensor([0])

        loss = criterion(outputs, targets)
        loss.backward()

        assert outputs.grad is not None, "Gradients should be computed"


class TestOptimizer:
    """Test optimizer functionality"""

    def test_adam_optimizer(self):
        """Test Adam optimizer"""
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Optimization step
        optimizer.zero_grad()
        outputs = model(torch.randn(4, 10))
        loss = outputs.sum()
        loss.backward()
        optimizer.step()

        # Check parameters changed
        for init, current in zip(initial_params, model.parameters()):
            assert not torch.equal(init, current), "Parameters should update"

    def test_learning_rate_scheduler(self):
        """Test LR scheduler"""
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        initial_lr = optimizer.param_groups[0]['lr']

        for _ in range(5):
            scheduler.step()

        new_lr = optimizer.param_groups[0]['lr']
        assert new_lr < initial_lr, "LR should decrease after step"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
