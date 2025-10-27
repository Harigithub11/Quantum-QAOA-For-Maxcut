"""
Integration tests for complete pipeline
"""

import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestEndToEndPipeline:
    """Test complete training and inference pipeline"""

    def test_data_to_model_pipeline(self):
        """Test data flows through model"""
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader

        # Load small subset
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        # Create small dataloader
        loader = DataLoader(dataset, batch_size=4)
        images, labels = next(iter(loader))

        assert images.shape == (4, 1, 28, 28)
        assert labels.shape == (4,)

    def test_model_training_step(self):
        """Test single training step"""
        try:
            from src.models.hybrid import HybridQuantumClassifier

            model = HybridQuantumClassifier(n_qubits=4, n_quantum_layers=1, n_classes=10)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            # Forward pass
            inputs = torch.randn(2, 1, 28, 28)
            targets = torch.tensor([3, 7])

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            assert loss.item() >= 0
        except ImportError:
            pytest.skip("Model not available")

    def test_model_inference(self):
        """Test model inference"""
        try:
            from src.models.hybrid import HybridQuantumClassifier

            model = HybridQuantumClassifier(n_qubits=4, n_quantum_layers=1, n_classes=10)
            model.eval()

            with torch.no_grad():
                inputs = torch.randn(4, 1, 28, 28)
                outputs = model(inputs)

                predictions = outputs.argmax(dim=1)
                assert predictions.shape == (4,)
                assert all(0 <= p < 10 for p in predictions)
        except ImportError:
            pytest.skip("Model not available")


class TestVisualizationPipeline:
    """Test visualization generation"""

    def test_matplotlib_import(self):
        """Test matplotlib is available"""
        import matplotlib.pyplot as plt
        assert plt is not None

    def test_plot_creation(self):
        """Test creating a simple plot"""
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)

        assert fig is not None
        plt.close(fig)


class TestGeminiIntegration:
    """Test Gemini API integration"""

    def test_gemini_client_creation(self):
        """Test creating Gemini client"""
        try:
            from src.gemini_integration import GeminiClient, MockGeminiClient

            # Test mock client
            client = MockGeminiClient()
            response = client.generate("Test prompt")
            assert response is not None
            assert isinstance(response, str)
        except ImportError:
            pytest.skip("Gemini integration not available")

    def test_prediction_explainer(self):
        """Test prediction explainer"""
        try:
            from src.gemini_integration import PredictionExplainer
            import numpy as np

            explainer = PredictionExplainer(use_mock=True)

            explanation = explainer.explain_prediction(
                image=np.random.rand(28, 28),
                prediction=7,
                confidence=0.95
            )

            assert explanation is not None
            assert isinstance(explanation, str)
        except ImportError:
            pytest.skip("Gemini integration not available")


class TestConfigurationLoading:
    """Test configuration file loading"""

    def test_config_files_exist(self):
        """Test config files exist"""
        config_dir = project_root / "configs"

        assert config_dir.exists(), "Config directory should exist"
        assert (config_dir / "model_config.yaml").exists(), "Model config should exist"
        assert (config_dir / "training_config.yaml").exists(), "Training config should exist"

    def test_load_yaml_config(self):
        """Test loading YAML configuration"""
        import yaml

        config_path = project_root / "configs" / "model_config.yaml"

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            assert config is not None
            assert isinstance(config, dict)


class TestCheckpointSaving:
    """Test model checkpoint saving and loading"""

    def test_save_checkpoint(self):
        """Test saving model checkpoint"""
        model = torch.nn.Linear(10, 2)
        checkpoint_path = project_root / "tests" / "test_checkpoint.pt"

        # Save
        torch.save(model.state_dict(), checkpoint_path)
        assert checkpoint_path.exists()

        # Clean up
        checkpoint_path.unlink()

    def test_load_checkpoint(self):
        """Test loading model checkpoint"""
        model = torch.nn.Linear(10, 2)
        checkpoint_path = project_root / "tests" / "test_checkpoint.pt"

        # Save
        torch.save(model.state_dict(), checkpoint_path)

        # Load
        new_model = torch.nn.Linear(10, 2)
        new_model.load_state_dict(torch.load(checkpoint_path))

        # Verify
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)

        # Clean up
        checkpoint_path.unlink()


class TestStreamlitApp:
    """Test Streamlit application"""

    def test_streamlit_files_exist(self):
        """Test Streamlit files exist"""
        app_dir = project_root / "app"

        assert app_dir.exists(), "App directory should exist"
        assert (app_dir / "streamlit_app.py").exists(), "Main app should exist"
        assert (app_dir / "pages").exists(), "Pages directory should exist"

    def test_all_pages_exist(self):
        """Test all 6 pages exist"""
        pages_dir = project_root / "app" / "pages"

        expected_pages = [
            "1_Train_Model.py",
            "2_Test_Model.py",
            "3_Visualizations.py",
            "4_Quantum_Analysis.py",
            "5_Experiments.py"
        ]

        for page in expected_pages:
            assert (pages_dir / page).exists(), f"Page {page} should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
