"""
Test Model Page
Make predictions with trained models and get Gemini explanations
"""

import streamlit as st
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set Gemini API key
os.environ['GEMINI_API_KEY'] = 'AIzaSyDmDbBs3N67TJsdPhQKbeOwO2cO72uEJ9k'

st.set_page_config(page_title="Test Model", page_icon="🎯", layout="wide")

st.title("🎯 Test Model")
st.markdown("Make predictions with your trained model and get AI-powered explanations.")

st.markdown("---")

# Model selection
st.markdown("### 🔧 Model Selection")

col1, col2 = st.columns([2, 1])

with col1:
    models_dir = project_root / "models" / "checkpoints"
    model_files = []

    if models_dir.exists():
        model_files = [f.name for f in models_dir.glob("*.pth")]

    if model_files:
        selected_model = st.selectbox(
            "Select Model Checkpoint",
            model_files,
            help="Choose a trained model checkpoint"
        )

        # Show checkpoint info
        import torch
        try:
            ckpt_path = models_dir / selected_model
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            metrics = checkpoint.get('metrics', {})
            st.info(f"📊 Epoch {checkpoint.get('epoch', 'N/A')} | Val Acc: {metrics.get('val_acc', 0):.2f}% | Val Loss: {metrics.get('val_loss', 0):.4f}")
        except:
            pass
    else:
        st.warning("⚠️ No trained models found. Please train a model first.")
        selected_model = None

with col2:
    use_gemini = st.checkbox(
        "🤖 Use Gemini Explanations",
        value=True,
        help="Get AI-powered explanations for predictions"
    )

    if use_gemini:
        st.success("✓ Gemini API configured")

st.markdown("---")

# Input method selection
st.markdown("### 📥 Input Method")

input_method = st.radio(
    "Choose input method:",
    ["Upload Image", "Select from Test Set", "Draw Digit (Coming Soon)"],
    horizontal=True
)

input_image = None
image_array = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a digit image (28x28 grayscale recommended)",
        type=["png", "jpg", "jpeg"],
        help="Upload an image of a handwritten digit"
    )

    if uploaded_file:
        input_image = Image.open(uploaded_file).convert('L')
        st.image(input_image, caption="Uploaded Image", width=200)

        # Resize to 28x28
        input_image = input_image.resize((28, 28))
        image_array = np.array(input_image) / 255.0

elif input_method == "Select from Test Set":
    st.info("Loading test set... (This will be implemented to load actual MNIST test images)")

    # Placeholder: generate random image
    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("🎲 Random Sample"):
            # In real implementation, load from actual test set
            image_array = np.random.rand(28, 28)
            st.session_state.test_image = image_array

    with col2:
        if 'test_image' in st.session_state:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(st.session_state.test_image, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
            image_array = st.session_state.test_image

elif input_method == "Draw Digit (Coming Soon)":
    st.info("📝 Drawing canvas feature coming soon! You'll be able to draw digits directly.")

st.markdown("---")

# Prediction section
if selected_model and image_array is not None:
    st.markdown("### 🔮 Make Prediction")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Run Prediction", type="primary"):
            with st.spinner("Running inference..."):
                # Mock prediction (replace with actual model inference)
                predicted_digit = np.random.randint(0, 10)
                confidence = np.random.uniform(0.7, 0.99)

                # Generate mock probability distribution
                probs = np.random.dirichlet(np.ones(10) * 0.5)
                probs[predicted_digit] = confidence
                probs = probs / probs.sum()

                st.session_state.prediction = predicted_digit
                st.session_state.confidence = confidence
                st.session_state.probs = probs

    with col2:
        if st.button("🔄 Clear Results"):
            if 'prediction' in st.session_state:
                del st.session_state.prediction
            if 'confidence' in st.session_state:
                del st.session_state.confidence

    # Display results
    if 'prediction' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Predicted Digit",
                value=st.session_state.prediction,
                delta=f"{st.session_state.confidence:.1%} confidence"
            )

        with col2:
            st.metric(
                label="Model Confidence",
                value=f"{st.session_state.confidence:.2%}"
            )

        with col3:
            certainty_level = "High" if st.session_state.confidence > 0.9 else "Medium" if st.session_state.confidence > 0.7 else "Low"
            st.metric(
                label="Certainty Level",
                value=certainty_level
            )

        # Probability distribution
        st.markdown("#### 📈 Class Probabilities")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(range(10), st.session_state.probs, color='steelblue', alpha=0.7)
            bars[st.session_state.prediction].set_color('orange')
            ax.set_xlabel('Digit', fontsize=12)
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
            ax.set_xticks(range(10))
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("**Top 3 Predictions:**")
            top_3_indices = np.argsort(st.session_state.probs)[-3:][::-1]
            for i, idx in enumerate(top_3_indices, 1):
                st.markdown(f"{i}. **Digit {idx}**: {st.session_state.probs[idx]:.2%}")

        # Gemini explanation
        if use_gemini:
            st.markdown("---")
            st.markdown("### 🤖 AI-Powered Explanation")

            with st.spinner("Generating explanation with Gemini..."):
                try:
                    from src.gemini_integration import PredictionExplainer

                    explainer = PredictionExplainer(use_mock=False)

                    # Get top-k probabilities
                    top_k_indices = np.argsort(st.session_state.probs)[-3:][::-1]
                    top_k_probs = {int(idx): float(st.session_state.probs[idx])
                                  for idx in top_k_indices}

                    explanation = explainer.explain_prediction(
                        image=image_array,
                        prediction=int(st.session_state.prediction),
                        confidence=float(st.session_state.confidence),
                        top_k_probs=top_k_probs
                    )

                    st.info(explanation)

                except Exception as e:
                    st.warning(f"Could not generate Gemini explanation: {e}")
                    st.info("""
                    **Mock Explanation:**

                    The model predicted this digit as **{}** with **{:.1%}** confidence.
                    This high confidence suggests that the quantum-classical hybrid network
                    successfully identified distinctive features in the input image.

                    The quantum circuit likely captured rotational and spatial patterns that
                    are characteristic of this digit, while the classical ResNet18 backbone
                    extracted robust visual features from the handwritten strokes.
                    """.format(st.session_state.prediction, st.session_state.confidence))

        # Grad-CAM visualization
        st.markdown("---")
        st.markdown("### 🔥 Grad-CAM Visualization")

        st.info("Grad-CAM heatmap will show which parts of the image the model focused on. (Coming after model training)")

        with st.expander("📖 What is Grad-CAM?"):
            st.markdown("""
            **Gradient-weighted Class Activation Mapping (Grad-CAM)** is a technique
            for visualizing which parts of an image are important for predictions.

            - **Red regions**: High importance
            - **Blue regions**: Low importance
            - Helps understand model decision-making
            - Useful for debugging and trust
            """)

else:
    st.info("👆 Select a model and provide an input image to make predictions.")

st.markdown("---")

# Batch prediction
st.markdown("### 📦 Batch Prediction")

with st.expander("Predict on Multiple Images"):
    st.markdown("""
    Upload multiple images to get batch predictions and analysis.

    Features:
    - Process multiple images at once
    - Compare predictions across samples
    - Get batch explanations from Gemini
    - Export results as CSV
    """)

    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if batch_files:
        st.info(f"Uploaded {len(batch_files)} images. Batch prediction feature coming soon!")

# Statistics
st.markdown("---")
st.markdown("### 📊 Prediction Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Predictions", "0", "No predictions yet")

with col2:
    st.metric("Average Confidence", "N/A", "Make predictions first")

with col3:
    st.metric("Most Common Digit", "N/A", "No data")

# Tips
with st.expander("💡 Tips for Better Predictions"):
    st.markdown("""
    ### Input Image Tips
    - Use 28×28 grayscale images for best results
    - Ensure digit is centered and fills most of the image
    - High contrast between digit and background works best
    - Avoid noisy or blurry images

    ### Understanding Confidence
    - **>90%**: Very confident prediction
    - **70-90%**: Confident, but check alternative predictions
    - **<70%**: Low confidence, may be ambiguous

    ### Using Gemini Explanations
    - Explanations help understand model reasoning
    - Useful for debugging misclassifications
    - Can reveal model biases or limitations
    - Combine with Grad-CAM for full picture
    """)
