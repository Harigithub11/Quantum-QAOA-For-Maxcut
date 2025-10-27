"""
Gemini Integration Module
Natural language explanations and automated report generation
"""

from .client import GeminiClient, MockGeminiClient, create_client, GEMINI_AVAILABLE

# Conditional imports
try:
    from .explainer import PredictionExplainer
    from .reporter import ReportGenerator
    EXPLAINER_AVAILABLE = True
except ImportError:
    EXPLAINER_AVAILABLE = False
    print("Warning: explainer and reporter modules not fully available")

__all__ = [
    'GeminiClient',
    'MockGeminiClient',
    'create_client',
    'GEMINI_AVAILABLE',
]

if EXPLAINER_AVAILABLE:
    __all__.extend(['PredictionExplainer', 'ReportGenerator'])


if __name__ == "__main__":
    print("Gemini Integration Package")
    print(f"- Gemini Available: {GEMINI_AVAILABLE}")
    print(f"- Explainer Available: {EXPLAINER_AVAILABLE}")
    print("\nModules:")
    print("  - client: GeminiClient for API interaction")
    print("  - explainer: PredictionExplainer for model explanations")
    print("  - reporter: ReportGenerator for automated reports")
