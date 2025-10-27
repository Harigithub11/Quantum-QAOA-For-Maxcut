"""
Test script to verify Phase 1 completion
"""

import sys
from pathlib import Path
import yaml
import os

# Set UTF-8 encoding for Windows console
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def test_directory_structure():
    """Test that all required directories exist."""
    print("Testing directory structure...")

    required_dirs = [
        "data/raw",
        "data/processed",
        "models/checkpoints",
        "notebooks",
        "src/data",
        "src/models",
        "src/training",
        "src/visualization",
        "src/gemini_integration",
        "app/pages",
        "configs",
        "tests",
        "logs",
        "mlruns",
        "results/figures",
        "results/reports"
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING!")
            all_exist = False

    return all_exist

def test_configuration_files():
    """Test that configuration files are valid."""
    print("\nTesting configuration files...")

    try:
        # Test model config
        with open("configs/model_config.yaml", 'r') as f:
            model_config = yaml.safe_load(f)
        print(f"  ✓ model_config.yaml loaded")
        print(f"    - Qubits: {model_config['quantum']['n_qubits']}")
        print(f"    - Layers: {model_config['quantum']['n_layers']}")

        # Test training config
        with open("configs/training_config.yaml", 'r') as f:
            training_config = yaml.safe_load(f)
        print(f"  ✓ training_config.yaml loaded")
        print(f"    - Epochs: {training_config['training']['num_epochs']}")
        print(f"    - Batch size: {training_config['data']['batch_size']}")

        return True
    except Exception as e:
        print(f"  ✗ Error loading configs: {e}")
        return False

def test_data_module():
    """Test the data module."""
    print("\nTesting data module...")

    try:
        from src.data.dataset import MNISTDataModule

        # Create data module (download=False to avoid downloading in test)
        data_module = MNISTDataModule(
            data_dir="./data/raw",
            batch_size=32
        )
        print(f"  ✓ MNISTDataModule imported successfully")
        print(f"  ✓ Data module initialized")

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_files_exist():
    """Test that all required files exist."""
    print("\nTesting required files...")

    required_files = [
        "requirements.txt",
        ".gitignore",
        ".env.example",
        "README.md",
        "setup_mlflow.py",
        "configs/model_config.yaml",
        "configs/training_config.yaml",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/data/dataset.py",
        "notebooks/01_data_exploration.ipynb"
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING!")
            all_exist = False

    return all_exist

def main():
    """Run all tests."""
    print("=" * 70)
    print("Phase 1: Project Setup & Data Pipeline - Verification")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Directory Structure", test_directory_structure()))
    results.append(("Configuration Files", test_configuration_files()))
    results.append(("Required Files", test_files_exist()))
    results.append(("Data Module", test_data_module()))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {test_name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 Phase 1 completed successfully!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Set up environment: cp .env.example .env (and add your API keys)")
        print("  3. Initialize MLflow: python setup_mlflow.py")
        print("  4. Test data pipeline: python -m src.data.dataset")
        print("  5. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
        print("\nReady to proceed to Phase 2: Quantum Circuit Implementation!")
    else:
        print("\n⚠ Some tests failed. Please review the errors above.")

    print("=" * 70)

if __name__ == "__main__":
    main()
