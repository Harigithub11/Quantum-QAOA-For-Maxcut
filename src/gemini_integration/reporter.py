"""
Automated Report Generator
Uses Gemini to create comprehensive project reports
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from .client import GeminiClient, create_client
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class ReportGenerator:
    """
    Generates comprehensive reports for quantum-classical ML projects.
    """

    def __init__(self, client: Optional[GeminiClient] = None, use_mock: bool = False):
        """
        Initialize report generator.

        Args:
            client: Gemini client
            use_mock: Use mock client
        """
        if client is None:
            self.client = create_client(use_mock=use_mock)
        else:
            self.client = client

    def generate_full_report(
        self,
        training_history: Dict[str, List[float]],
        test_metrics: Dict[str, float],
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        per_class_metrics: Optional[Dict[int, Dict[str, float]]] = None,
        quantum_stats: Optional[Dict[str, Any]] = None,
        save_path: Optional[Path] = None
    ) -> str:
        """
        Generate comprehensive project report.

        Args:
            training_history: Training metrics history
            test_metrics: Test set metrics
            model_config: Model configuration
            training_config: Training configuration
            per_class_metrics: Per-class performance metrics
            quantum_stats: Quantum layer statistics
            save_path: Path to save report

        Returns:
            Full report as markdown string
        """
        print("Generating comprehensive report...")

        # Generate each section
        executive_summary = self._generate_executive_summary(
            test_metrics, training_history
        )

        methodology = self._generate_methodology(
            model_config, training_config
        )

        results = self._generate_results_section(
            training_history, test_metrics, per_class_metrics
        )

        quantum_analysis = self._generate_quantum_analysis(
            quantum_stats
        ) if quantum_stats else ""

        conclusions = self._generate_conclusions(
            test_metrics, training_history
        )

        # Compile report
        report = self._compile_report(
            executive_summary=executive_summary,
            methodology=methodology,
            results=results,
            quantum_analysis=quantum_analysis,
            conclusions=conclusions
        )

        # Save if path provided
        if save_path:
            self._save_report(report, save_path)

        print("Report generation complete!")
        return report

    def _generate_executive_summary(
        self,
        test_metrics: Dict[str, float],
        training_history: Dict[str, List[float]]
    ) -> str:
        """Generate executive summary section."""
        prompt = f"""Generate an executive summary (2-3 paragraphs) for a hybrid quantum-classical machine learning project.

Key Results:
- Test Accuracy: {test_metrics.get('accuracy', 0.0):.2%}
- Test Loss: {test_metrics.get('loss', 0.0):.4f}
- Training Epochs: {len(training_history.get('train_loss', []))}
- Best Validation Accuracy: {max(training_history.get('val_acc', [0])):.2%}

Context:
- Task: MNIST digit classification (10 classes)
- Architecture: Hybrid quantum-classical neural network
- Classical component: ResNet18 (pretrained)
- Quantum component: 4-qubit variational circuit

Write a concise, professional executive summary covering:
1. Project objective
2. Key achievements
3. Main findings

Use clear, accessible language."""

        try:
            return self.client.generate(prompt)
        except Exception as e:
            return f"[Executive summary unavailable: {e}]"

    def _generate_methodology(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> str:
        """Generate methodology section."""
        prompt = f"""Generate a methodology section (2-3 paragraphs) describing the technical approach.

Model Architecture:
{json.dumps(model_config, indent=2)}

Training Configuration:
{json.dumps(training_config, indent=2)}

Write a clear methodology covering:
1. Classical feature extraction approach
2. Quantum circuit design
3. Training procedure and hyperparameters
4. Evaluation methodology

Use technical but accessible language."""

        try:
            return self.client.generate(prompt)
        except Exception as e:
            return f"[Methodology unavailable: {e}]"

    def _generate_results_section(
        self,
        training_history: Dict[str, List[float]],
        test_metrics: Dict[str, float],
        per_class_metrics: Optional[Dict[int, Dict[str, float]]]
    ) -> str:
        """Generate results section."""
        prompt = f"""Generate a results section (3-4 paragraphs) analyzing the model's performance.

Training History:
- Epochs: {len(training_history.get('train_loss', []))}
- Final Training Loss: {training_history.get('train_loss', [0])[-1]:.4f}
- Final Validation Loss: {training_history.get('val_loss', [0])[-1]:.4f}
- Final Training Accuracy: {training_history.get('train_acc', [0])[-1]:.2f}%
- Final Validation Accuracy: {training_history.get('val_acc', [0])[-1]:.2f}%

Test Set Performance:
- Accuracy: {test_metrics.get('accuracy', 0.0):.2%}
- Loss: {test_metrics.get('loss', 0.0):.4f}
"""

        if per_class_metrics:
            prompt += "\nPer-Class Accuracy:\n"
            for digit, metrics in sorted(per_class_metrics.items()):
                acc = metrics.get('accuracy', 0.0)
                prompt += f"- Digit {digit}: {acc:.2%}\n"

        prompt += """
Write a comprehensive results section covering:
1. Training progression and convergence
2. Test set performance
3. Per-class performance strengths and weaknesses
4. Comparison to baseline expectations

Use data-driven analysis."""

        try:
            return self.client.generate(prompt)
        except Exception as e:
            return f"[Results unavailable: {e}]"

    def _generate_quantum_analysis(
        self,
        quantum_stats: Dict[str, Any]
    ) -> str:
        """Generate quantum circuit analysis section."""
        prompt = f"""Generate a quantum circuit analysis section (2-3 paragraphs).

Quantum Layer Statistics:
{json.dumps(quantum_stats, indent=2)}

Write analysis covering:
1. Quantum circuit contribution to model performance
2. Parameter evolution during training
3. Quantum advantages or limitations observed
4. Entanglement and quantum state characteristics

Use quantum computing terminology appropriately."""

        try:
            return self.client.generate(prompt)
        except Exception as e:
            return f"[Quantum analysis unavailable: {e}]"

    def _generate_conclusions(
        self,
        test_metrics: Dict[str, float],
        training_history: Dict[str, List[float]]
    ) -> str:
        """Generate conclusions and future work section."""
        prompt = f"""Generate conclusions and recommendations (2-3 paragraphs).

Project Outcomes:
- Achieved Test Accuracy: {test_metrics.get('accuracy', 0.0):.2%}
- Training Stability: {"Stable" if len(training_history.get('train_loss', [])) > 0 else "Unknown"}

Write conclusions covering:
1. Summary of achievements
2. Limitations and challenges
3. Recommendations for future improvements
4. Potential applications or extensions

Be forward-looking and constructive."""

        try:
            return self.client.generate(prompt)
        except Exception as e:
            return f"[Conclusions unavailable: {e}]"

    def _compile_report(
        self,
        executive_summary: str,
        methodology: str,
        results: str,
        quantum_analysis: str,
        conclusions: str
    ) -> str:
        """Compile all sections into final report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""# Hybrid Quantum-Classical Machine Learning Project Report

**Generated**: {timestamp}

---

## Executive Summary

{executive_summary}

---

## Methodology

{methodology}

---

## Results and Analysis

{results}

"""

        if quantum_analysis:
            report += f"""---

## Quantum Circuit Analysis

{quantum_analysis}

"""

        report += f"""---

## Conclusions and Future Work

{conclusions}

---

## Technical Details

### Model Architecture
- **Classical Component**: ResNet18 (pretrained on ImageNet)
  - Input: 28x28 grayscale images
  - Feature dimension: Configurable (default: 4)
  - Frozen layers for transfer learning

- **Quantum Component**: Variational Quantum Circuit
  - Qubits: 4
  - Layers: 3 (default)
  - Device: PennyLane default.qubit simulator
  - Differentiation: Parameter-shift rule

- **Classifier**: Fully connected layer
  - Input: Quantum output features
  - Output: 10 classes (digits 0-9)

### Training Infrastructure
- Framework: PyTorch + PennyLane
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Device: CPU/CUDA (configurable)
- Experiment tracking: MLflow + TensorBoard

### Evaluation Metrics
- Accuracy (overall and per-class)
- Confusion matrix
- Precision, recall, F1-score
- Training/validation loss curves

---

*Report generated with Claude Code - Quantum-Classical ML Project*
"""

        return report

    def _save_report(self, report: str, save_path: Path) -> None:
        """Save report to file."""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save markdown
        md_path = save_path.with_suffix('.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Report saved to {md_path}")

        # Optionally save as text
        txt_path = save_path.with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Report also saved as {txt_path}")

    def generate_training_summary(
        self,
        training_history: Dict[str, List[float]],
        save_path: Optional[Path] = None
    ) -> str:
        """
        Generate quick training summary.

        Args:
            training_history: Training history
            save_path: Path to save

        Returns:
            Summary string
        """
        epochs = len(training_history.get('train_loss', []))

        summary = f"""# Training Summary

**Total Epochs**: {epochs}

## Final Metrics
- Training Loss: {training_history.get('train_loss', [0])[-1]:.4f}
- Training Accuracy: {training_history.get('train_acc', [0])[-1]:.2f}%
- Validation Loss: {training_history.get('val_loss', [0])[-1]:.4f}
- Validation Accuracy: {training_history.get('val_acc', [0])[-1]:.2f}%

## Best Performance
- Best Validation Accuracy: {max(training_history.get('val_acc', [0])):.2f}%
- Best Validation Loss: {min(training_history.get('val_loss', [float('inf')])):.4f}
- Epoch of Best Val Acc: {training_history.get('val_acc', [0]).index(max(training_history.get('val_acc', [0]))) + 1}

## Training Dynamics
- Initial Loss: {training_history.get('train_loss', [0])[0]:.4f}
- Final Loss: {training_history.get('train_loss', [0])[-1]:.4f}
- Loss Reduction: {((training_history.get('train_loss', [1])[0] - training_history.get('train_loss', [1])[-1]) / training_history.get('train_loss', [1])[0] * 100):.1f}%
"""

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"Training summary saved to {save_path}")

        return summary


if __name__ == "__main__":
    print("Testing Report Generator...")

    # Test with mock client
    reporter = ReportGenerator(use_mock=True)

    # Dummy data
    training_history = {
        'train_loss': [2.3, 2.1, 1.8, 1.5, 1.2],
        'val_loss': [2.4, 2.2, 1.9, 1.7, 1.5],
        'train_acc': [20, 35, 50, 65, 75],
        'val_acc': [18, 32, 48, 62, 72],
    }

    test_metrics = {
        'accuracy': 0.72,
        'loss': 1.6
    }

    model_config = {
        'classical': {'model_name': 'resnet18', 'pretrained': True},
        'quantum': {'n_qubits': 4, 'n_layers': 3}
    }

    training_config = {
        'batch_size': 64,
        'learning_rate': 0.001,
        'num_epochs': 5
    }

    # Generate report
    report = reporter.generate_full_report(
        training_history=training_history,
        test_metrics=test_metrics,
        model_config=model_config,
        training_config=training_config,
        save_path=Path('test_report.md')
    )

    print(f"\nGenerated report ({len(report)} characters)")
    print("Report Generator module ready!")
