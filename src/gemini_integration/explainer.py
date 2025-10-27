"""
Gemini-powered Prediction Explainer
Generates natural language explanations for model predictions
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

try:
    from .client import GeminiClient, create_client
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class PredictionExplainer:
    """
    Generates explanations for quantum-classical hybrid model predictions.
    """

    def __init__(self, client: Optional[GeminiClient] = None, use_mock: bool = False):
        """
        Initialize explainer.

        Args:
            client: Gemini client (or None to create default)
            use_mock: Use mock client for testing
        """
        if client is None:
            self.client = create_client(use_mock=use_mock)
        else:
            self.client = client

        self.explanation_cache = {}

    def explain_prediction(
        self,
        image: np.ndarray,
        prediction: int,
        confidence: float,
        true_label: Optional[int] = None,
        quantum_contribution: Optional[float] = None,
        top_k_probs: Optional[Dict[int, float]] = None
    ) -> str:
        """
        Explain a single prediction.

        Args:
            image: Input image (28, 28)
            prediction: Predicted class
            confidence: Prediction confidence (0-1)
            true_label: True label (if available)
            quantum_contribution: Quantum layer contribution
            top_k_probs: Top-k class probabilities

        Returns:
            Natural language explanation
        """
        # Build context
        context = self._build_context(
            prediction=prediction,
            confidence=confidence,
            true_label=true_label,
            quantum_contribution=quantum_contribution,
            top_k_probs=top_k_probs
        )

        # Create prompt
        prompt = f"""You are an AI assistant explaining predictions from a hybrid quantum-classical neural network for MNIST digit classification.

Architecture:
- Classical Feature Extractor: ResNet18 (pretrained on ImageNet)
- Quantum Layer: 4-qubit variational quantum circuit with parameter-shift gradients
- Classifier: Fully connected output layer (10 classes)

Prediction Details:
{context}

Task: Provide a clear, concise explanation (2-3 sentences) of why the model made this prediction. Consider:
1. The confidence level
2. The quantum circuit's contribution (if available)
3. Whether the prediction is correct
4. Any competing predictions

Keep the explanation accessible to non-experts."""

        try:
            explanation = self.client.generate(prompt)
            return explanation.strip()
        except Exception as e:
            return f"[Explanation unavailable: {str(e)}]"

    def explain_batch(
        self,
        predictions: List[int],
        confidences: List[float],
        true_labels: Optional[List[int]] = None,
        quantum_contributions: Optional[List[float]] = None,
        num_explain: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Explain multiple predictions.

        Args:
            predictions: List of predicted classes
            confidences: List of confidence scores
            true_labels: List of true labels
            quantum_contributions: List of quantum contributions
            num_explain: Number to explain (most interesting cases)

        Returns:
            List of explanation dictionaries
        """
        # Select interesting cases
        indices = self._select_interesting_cases(
            predictions=predictions,
            confidences=confidences,
            true_labels=true_labels,
            num_select=num_explain
        )

        explanations = []
        for idx in indices:
            explanation = self.explain_prediction(
                image=None,  # Not needed for text explanation
                prediction=predictions[idx],
                confidence=confidences[idx],
                true_label=true_labels[idx] if true_labels else None,
                quantum_contribution=quantum_contributions[idx] if quantum_contributions else None
            )

            explanations.append({
                'index': int(idx),
                'prediction': int(predictions[idx]),
                'confidence': float(confidences[idx]),
                'true_label': int(true_labels[idx]) if true_labels else None,
                'explanation': explanation
            })

        return explanations

    def explain_confusion(
        self,
        confusion_pairs: List[Tuple[int, int]],
        frequencies: List[int]
    ) -> Dict[str, str]:
        """
        Explain common confusion patterns.

        Args:
            confusion_pairs: List of (true_label, predicted_label) tuples
            frequencies: Frequency of each confusion

        Returns:
            Dictionary of explanations
        """
        prompt = f"""You are analyzing confusion patterns in a hybrid quantum-classical MNIST classifier.

Common Confusion Patterns:
"""
        for (true_label, pred_label), freq in zip(confusion_pairs, frequencies):
            prompt += f"- True: {true_label}, Predicted: {pred_label}, Count: {freq}\n"

        prompt += """
Task: For each confusion pattern, explain in 1-2 sentences why the model might confuse these digits. Consider:
1. Visual similarity between digits
2. Potential limitations of the quantum circuit
3. Feature extraction challenges

Format your response as:
[True digit] -> [Predicted digit]: [Explanation]
"""

        try:
            response = self.client.generate(prompt)
            explanations = self._parse_confusion_explanations(response)
            return explanations
        except Exception as e:
            return {f"{true}->{pred}": f"[Unavailable: {e}]"
                    for true, pred in confusion_pairs}

    def analyze_model_behavior(
        self,
        accuracy: float,
        per_class_accuracy: Dict[int, float],
        quantum_layer_stats: Optional[Dict[str, float]] = None,
        training_history: Optional[Dict[str, List[float]]] = None
    ) -> str:
        """
        Generate comprehensive model behavior analysis.

        Args:
            accuracy: Overall accuracy
            per_class_accuracy: Accuracy per digit class
            quantum_layer_stats: Quantum layer statistics
            training_history: Training history

        Returns:
            Detailed analysis report
        """
        prompt = f"""You are analyzing the performance of a hybrid quantum-classical neural network for MNIST classification.

Overall Performance:
- Test Accuracy: {accuracy:.2%}

Per-Class Accuracy:
"""
        for digit, acc in sorted(per_class_accuracy.items()):
            prompt += f"- Digit {digit}: {acc:.2%}\n"

        if quantum_layer_stats:
            prompt += f"\nQuantum Layer Statistics:\n"
            for key, value in quantum_layer_stats.items():
                prompt += f"- {key}: {value}\n"

        if training_history:
            best_val_acc = max(training_history.get('val_acc', [0]))
            prompt += f"\nTraining:\n"
            prompt += f"- Best Validation Accuracy: {best_val_acc:.2%}\n"
            prompt += f"- Total Epochs: {len(training_history.get('train_loss', []))}\n"

        prompt += """
Task: Provide a comprehensive analysis (3-4 paragraphs) covering:
1. Overall model performance assessment
2. Strengths and weaknesses by digit class
3. Quantum layer's contribution to performance
4. Potential improvements or next steps

Write in a clear, professional style suitable for a technical report."""

        try:
            analysis = self.client.generate(prompt)
            return analysis.strip()
        except Exception as e:
            return f"[Analysis unavailable: {str(e)}]"

    def _build_context(
        self,
        prediction: int,
        confidence: float,
        true_label: Optional[int],
        quantum_contribution: Optional[float],
        top_k_probs: Optional[Dict[int, float]]
    ) -> str:
        """Build context string for prompt."""
        context = f"- Predicted Digit: {prediction}\n"
        context += f"- Confidence: {confidence:.2%}\n"

        if true_label is not None:
            correct = "CORRECT" if prediction == true_label else "INCORRECT"
            context += f"- True Label: {true_label} ({correct})\n"

        if quantum_contribution is not None:
            context += f"- Quantum Layer Contribution: {quantum_contribution:.3f}\n"

        if top_k_probs:
            context += "- Top Predictions:\n"
            for digit, prob in sorted(top_k_probs.items(), key=lambda x: x[1], reverse=True):
                context += f"  - Digit {digit}: {prob:.2%}\n"

        return context

    def _select_interesting_cases(
        self,
        predictions: List[int],
        confidences: List[float],
        true_labels: Optional[List[int]],
        num_select: int
    ) -> List[int]:
        """Select most interesting cases for explanation."""
        scores = []

        for i in range(len(predictions)):
            score = 0.0

            # Prioritize incorrect predictions
            if true_labels and predictions[i] != true_labels[i]:
                score += 2.0

            # Prioritize low confidence
            score += (1.0 - confidences[i])

            # Prioritize medium confidence (uncertainty)
            if 0.3 < confidences[i] < 0.7:
                score += 0.5

            scores.append((i, score))

        # Sort by score and select top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        indices = [idx for idx, _ in scores[:num_select]]

        return indices

    def _parse_confusion_explanations(self, response: str) -> Dict[str, str]:
        """Parse confusion explanations from Gemini response."""
        explanations = {}

        for line in response.strip().split('\n'):
            if '->' in line:
                try:
                    # Parse "True -> Pred: Explanation"
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        explanation = parts[1].strip()
                        explanations[key] = explanation
                except:
                    continue

        return explanations

    def save_explanations(
        self,
        explanations: List[Dict[str, Any]],
        save_path: Path
    ) -> None:
        """
        Save explanations to JSON file.

        Args:
            explanations: List of explanation dictionaries
            save_path: Path to save
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(explanations, f, indent=2)

        print(f"Explanations saved to {save_path}")


if __name__ == "__main__":
    print("Testing Prediction Explainer...")

    # Test with mock client
    explainer = PredictionExplainer(use_mock=True)

    # Test single prediction
    explanation = explainer.explain_prediction(
        image=None,
        prediction=7,
        confidence=0.92,
        true_label=7,
        quantum_contribution=0.15,
        top_k_probs={7: 0.92, 1: 0.05, 9: 0.02}
    )
    print(f"Single prediction explanation:\n{explanation}\n")

    # Test batch
    explanations = explainer.explain_batch(
        predictions=[7, 3, 5, 8, 1],
        confidences=[0.92, 0.65, 0.88, 0.45, 0.78],
        true_labels=[7, 8, 5, 8, 1],
        num_explain=3
    )
    print(f"Batch explanations: {len(explanations)} generated\n")

    # Test confusion analysis
    confusion_explanations = explainer.explain_confusion(
        confusion_pairs=[(3, 8), (5, 3), (7, 1)],
        frequencies=[12, 8, 6]
    )
    print(f"Confusion analysis: {len(confusion_explanations)} patterns\n")

    print("Prediction Explainer module ready!")
