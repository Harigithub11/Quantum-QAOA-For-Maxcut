"""
MLflow Setup and Initialization Script
"""

import mlflow
from pathlib import Path
import os


def setup_mlflow():
    """
    Initialize MLflow tracking and create experiment.
    """
    # Set tracking URI to local directory
    mlflow_dir = Path("./mlruns")
    mlflow_dir.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")

    # Create or get experiment
    experiment_name = "quantum-mnist-hybrid"

    try:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={
                "project": "Quantum ML MNIST",
                "framework": "PyTorch + PennyLane",
                "model_type": "Hybrid Quantum-Classical"
            }
        )
        print(f"✓ Created new experiment: {experiment_name}")
        print(f"  Experiment ID: {experiment_id}")
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"✓ Using existing experiment: {experiment_name}")
        print(f"  Experiment ID: {experiment_id}")

    # Set experiment as active
    mlflow.set_experiment(experiment_name)

    print(f"\n✓ MLflow setup complete!")
    print(f"  Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"  Active experiment: {experiment_name}")
    print(f"\nTo view MLflow UI, run:")
    print(f"  mlflow ui --backend-store-uri {mlflow_dir.absolute()}")

    return experiment_id


def test_mlflow():
    """
    Test MLflow logging with a dummy run.
    """
    print("\nTesting MLflow logging...")

    with mlflow.start_run(run_name="test_run"):
        # Log parameters
        mlflow.log_param("test_param", "test_value")
        mlflow.log_param("batch_size", 64)

        # Log metrics
        mlflow.log_metric("test_accuracy", 0.95)
        mlflow.log_metric("test_loss", 0.15)

        # Log tags
        mlflow.set_tag("test", "true")

        run_id = mlflow.active_run().info.run_id
        print(f"✓ Test run created successfully!")
        print(f"  Run ID: {run_id}")

    print(f"\n✓ MLflow logging test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("MLflow Setup for Quantum ML MNIST Project")
    print("=" * 60)

    # Setup MLflow
    experiment_id = setup_mlflow()

    # Test MLflow
    test_mlflow()

    print("\n" + "=" * 60)
    print("Setup complete! Ready to track experiments.")
    print("=" * 60)
