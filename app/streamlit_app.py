"""
Hybrid Neural Networks Dashboard
Main Streamlit application with multi-page navigation
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="Hybrid Neural Networks for MNIST Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/FFFFFF?text=Hybrid+NN")

    st.markdown("---")

    st.markdown("### 🔬 Project Info")
    st.markdown("""
    **Hybrid Neural Networks**
    **for MNIST Classification**

    - **Classical**: ResNet18 (pretrained)
    - **Quantum**: 4-qubit VQC
    - **Task**: Digit classification (0-9)
    """)

    st.markdown("---")

    st.markdown("### 📊 Navigation")
    st.markdown("""
    Use the pages in the sidebar to:
    - **Train Model**: Configure and train
    - **Test Model**: Make predictions
    - **Visualizations**: Explore results
    - **Quantum Analysis**: Dive into quantum
    - **Experiments**: Compare runs
    """)

    st.markdown("---")

    st.markdown("### ⚙️ Quick Actions")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    if st.button("📥 Download Models"):
        st.info("Model download feature coming soon!")

    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("""
    - [Documentation](README.md)
    - [GitHub Repo](#)
    - [Paper (Coming Soon)](#)
    """)

# Main page content
st.markdown('<p class="main-header">✂️ Adaptive QAOA for MaxCut Problem</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Noise-Aware, Warm-Started Quantum Approximate Optimization</p>',
            unsafe_allow_html=True)

# Welcome section
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Objective")
    st.markdown("""
    Solve the NP-hard Maximum Cut problem using quantum computing with adaptive
    circuit construction, classical warm-starting, and noise-aware optimization.
    """)

with col2:
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    - **Classical Pre-solver**: Greedy heuristic for warm-starting
    - **Quantum Circuit**: QAOA with adaptive ansatz growth
    - **Hybrid Loop**: Classical optimization + quantum evaluation
    """)

with col3:
    st.markdown("### 📈 Performance")
    st.markdown("""
    - **Dataset**: MNIST (70,000 images)
    - **Classes**: 10 digits (0-9)
    - **Target**: >95% accuracy
    """)

st.markdown("---")

# System status
st.markdown("### 🖥️ System Status")

# Load checkpoint data
import torch
checkpoints_dir = project_root / "models" / "checkpoints"
checkpoint_files = []
if checkpoints_dir.exists():
    checkpoint_files = sorted(list(checkpoints_dir.glob("checkpoint_epoch_*.pth")))

# Get latest metrics
models_trained = len(checkpoint_files)
best_val_acc = "N/A"
latest_epoch = "N/A"

if checkpoint_files:
    try:
        # Load latest checkpoint
        latest_ckpt = torch.load(checkpoint_files[-1], map_location='cpu')
        metrics = latest_ckpt.get('metrics', {})
        best_val_acc = f"{metrics.get('val_acc', 0):.2f}%"
        latest_epoch = f"Epoch {latest_ckpt.get('epoch', 0)}"

        # Find best validation accuracy across all checkpoints
        best_acc = 0
        for ckpt_file in checkpoint_files:
            ckpt = torch.load(ckpt_file, map_location='cpu')
            val_acc = ckpt.get('metrics', {}).get('val_acc', 0)
            if val_acc > best_acc:
                best_acc = val_acc
        best_val_acc = f"{best_acc:.2f}%"
    except Exception as e:
        best_val_acc = "Error loading"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Models Trained",
        value=str(models_trained),
        delta="Checkpoints saved" if models_trained > 0 else "Ready to train"
    )

with col2:
    st.metric(
        label="Best Val Accuracy",
        value=best_val_acc,
        delta=latest_epoch if models_trained > 0 else "Train first"
    )

with col3:
    st.metric(
        label="Quantum Qubits",
        value="4",
        delta="Configured"
    )

with col4:
    st.metric(
        label="Total Parameters",
        value="70,182",
        delta="Trainable"
    )

st.markdown("---")

# Quick start guide
st.markdown("### 🚀 Quick Start Guide")

tab1, tab2, tab3 = st.tabs(["📖 Getting Started", "🔬 How It Works", "💡 Tips"])

with tab1:
    st.markdown("""
    #### Getting Started with Quantum-Classical ML

    1. **Train a Model** 📚
       - Navigate to "Train Model" page
       - Configure hyperparameters
       - Start training and monitor progress

    2. **Test Predictions** 🎯
       - Go to "Test Model" page
       - Upload or draw a digit
       - Get predictions with explanations

    3. **Explore Visualizations** 📊
       - Check "Visualizations" page
       - View training curves, confusion matrices
       - Explore t-SNE embeddings

    4. **Analyze Quantum Behavior** ⚛️
       - Visit "Quantum Analysis" page
       - Understand quantum circuit contribution
       - Visualize quantum states

    5. **Compare Experiments** 🔬
       - Use "Experiments" page
       - Compare different configurations
       - Track hyperparameter impact
    """)

with tab2:
    st.markdown("""
    #### How the Hybrid Model Works

    **Classical Component (ResNet18)**
    - Pre-trained on ImageNet for robust feature extraction
    - Processes 28×28 grayscale MNIST images
    - Extracts 4-dimensional feature vectors
    - Frozen layers for transfer learning

    **Quantum Component (VQC)**
    - 4 qubits encode the classical features
    - 3 variational layers with trainable parameters
    - Uses PennyLane's parameter-shift gradients
    - Outputs quantum measurement results

    **Classifier**
    - Fully connected layer
    - Maps quantum outputs to 10 digit classes
    - Trained end-to-end with backpropagation

    **Training Pipeline**
    1. Image → ResNet18 → 4D features
    2. Features → Quantum Circuit → Quantum output
    3. Quantum output → Classifier → Predictions
    4. Loss calculation and backpropagation
    5. Parameter updates (classical + quantum)
    """)

with tab3:
    st.markdown("""
    #### Tips for Best Results

    **Training Tips**
    - Start with default hyperparameters
    - Use batch size 64-128 for stability
    - Monitor both train and validation metrics
    - Enable early stopping to prevent overfitting

    **Quantum Circuit Tips**
    - More qubits ≠ better performance (4 is optimal)
    - 2-3 quantum layers work well
    - Too many layers can cause barren plateaus

    **Performance Tips**
    - CPU training is faster for this model (quantum bottleneck)
    - Use GPU only if quantum backend supports it
    - Cache data for faster loading

    **Visualization Tips**
    - t-SNE takes time - use subset of data
    - Grad-CAM shows what model focuses on
    - Confusion matrix reveals weak points

    **Experiment Tracking**
    - MLflow automatically logs all experiments
    - Compare runs to find best configuration
    - Export results for presentations
    """)

st.markdown("---")

# Architecture diagram
st.markdown("### 🏗️ Model Architecture")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────┐
    │            Input Image (28×28)              │
    └──────────────────┬──────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────┐
    │     ResNet18 Feature Extractor (Frozen)     │
    │     • Pre-trained on ImageNet               │
    │     • Outputs 4D feature vector             │
    └──────────────────┬──────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────┐
    │     Quantum Circuit (4 qubits, 3 layers)    │
    │     • Variational quantum circuit           │
    │     • 8 trainable parameters                │
    └──────────────────┬──────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────┐
    │         Classifier (Fully Connected)        │
    │     • Maps to 10 output classes             │
    └──────────────────┬──────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────┐
    │          Predictions (0-9 digits)           │
    └─────────────────────────────────────────────┘
    ```
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem;">
    <p>Built with Streamlit • PyTorch • PennyLane</p>
    <p>Quantum-Classical Machine Learning Project • 2024</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'training_active' not in st.session_state:
    st.session_state.training_active = False
if 'history' not in st.session_state:
    st.session_state.history = None
