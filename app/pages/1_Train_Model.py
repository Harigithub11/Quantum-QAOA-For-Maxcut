"""
Train Model Page
Configure hyperparameters and start training
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Train Model", page_icon="📚", layout="wide")

st.title("📚 Train Model")
st.markdown("Configure hyperparameters and start training your hybrid quantum-classical model.")

st.markdown("---")

# Configuration section
st.markdown("### ⚙️ Model Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Classical Component")

    classical_model = st.selectbox(
        "Backbone Model",
        ["resnet18", "resnet34", "resnet50"],
        index=0,
        help="Pre-trained feature extractor"
    )

    pretrained = st.checkbox(
        "Use Pre-trained Weights",
        value=True,
        help="Load ImageNet pre-trained weights"
    )

    freeze_layers = st.checkbox(
        "Freeze Backbone",
        value=True,
        help="Freeze classical layers for transfer learning"
    )

    feature_dim = st.slider(
        "Feature Dimension",
        min_value=2,
        max_value=8,
        value=4,
        help="Dimension of classical feature output"
    )

with col2:
    st.markdown("#### Quantum Component")

    n_qubits = st.slider(
        "Number of Qubits",
        min_value=2,
        max_value=8,
        value=4,
        help="Must match feature dimension"
    )

    n_quantum_layers = st.slider(
        "Quantum Layers",
        min_value=1,
        max_value=5,
        value=2,
        help="Number of variational layers"
    )

    quantum_device = st.selectbox(
        "Quantum Device",
        ["default.qubit", "default.mixed", "lightning.qubit"],
        index=0,
        help="PennyLane quantum device"
    )

    diff_method = st.selectbox(
        "Differentiation Method",
        ["parameter-shift", "backprop", "adjoint"],
        index=0,
        help="Gradient computation method"
    )

# Validate configuration
if feature_dim != n_qubits:
    st.warning(f"⚠️ Feature dimension ({feature_dim}) should match number of qubits ({n_qubits})")

st.markdown("---")

# Training configuration
st.markdown("### 🎯 Training Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Data")

    batch_size = st.select_slider(
        "Batch Size",
        options=[16, 32, 64, 128, 256],
        value=64,
        help="Larger batch = faster but more memory"
    )

    num_workers = st.slider(
        "Data Workers",
        min_value=0,
        max_value=8,
        value=4,
        help="Number of data loading workers"
    )

    pin_memory = st.checkbox(
        "Pin Memory",
        value=False,
        help="Enable for GPU training"
    )

with col2:
    st.markdown("#### Optimization")

    num_epochs = st.slider(
        "Number of Epochs",
        min_value=1,
        max_value=50,
        value=5,
        help="Training epochs"
    )

    learning_rate = st.select_slider(
        "Learning Rate",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
        value=0.001,
        help="Optimizer learning rate"
    )

    optimizer = st.selectbox(
        "Optimizer",
        ["Adam", "SGD", "AdamW"],
        index=0
    )

    use_scheduler = st.checkbox(
        "Use LR Scheduler",
        value=True,
        help="Step LR scheduler"
    )

with col3:
    st.markdown("#### Regularization")

    weight_decay = st.select_slider(
        "Weight Decay",
        options=[0.0, 1e-5, 1e-4, 1e-3],
        value=1e-4,
        help="L2 regularization"
    )

    gradient_clip = st.slider(
        "Gradient Clipping",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        help="Max gradient norm"
    )

    early_stopping = st.checkbox(
        "Early Stopping",
        value=True,
        help="Stop if no improvement"
    )

    if early_stopping:
        patience = st.slider(
            "Patience",
            min_value=3,
            max_value=20,
            value=10,
            help="Epochs to wait"
        )

st.markdown("---")

# Device configuration
st.markdown("### 🖥️ Device Configuration")

col1, col2 = st.columns(2)

with col1:
    use_cuda = st.checkbox(
        "Use CUDA (GPU)",
        value=False,
        help="Use GPU for training (CPU often faster for hybrid models)"
    )

    use_amp = st.checkbox(
        "Mixed Precision Training",
        value=False,
        disabled=not use_cuda,
        help="Enable only with CUDA"
    )

with col2:
    st.info("""
    **💡 Tip**: CPU training is often faster for hybrid quantum-classical models
    due to quantum circuit computation bottleneck on CPU.
    """)

st.markdown("---")

# Experiment tracking
st.markdown("### 📊 Experiment Tracking")

col1, col2 = st.columns(2)

with col1:
    experiment_name = st.text_input(
        "Experiment Name",
        value=f"hybrid_q{n_qubits}l{n_quantum_layers}",
        help="MLflow experiment name"
    )

    run_name = st.text_input(
        "Run Name",
        value=f"run_{time.strftime('%Y%m%d_%H%M%S')}",
        help="MLflow run name"
    )

with col2:
    log_interval = st.slider(
        "Log Interval",
        min_value=1,
        max_value=100,
        value=10,
        help="Log metrics every N batches"
    )

    save_interval = st.slider(
        "Save Interval",
        min_value=1,
        max_value=10,
        value=1,
        help="Save checkpoint every N epochs"
    )

st.markdown("---")

# Build command
st.markdown("### 🚀 Training Command")

command_args = [
    f"--batch-size {batch_size}",
    f"--epochs {num_epochs}",
    f"--learning-rate {learning_rate}",
    f"--quantum-layers {n_quantum_layers}",
]

if not use_cuda:
    command_args.append("--no-cuda")
if use_amp:
    command_args.append("--use-amp")
if early_stopping:
    command_args.append(f"--early-stopping-patience {patience}")

command = f"python train.py {' '.join(command_args)}"

st.code(command, language="bash")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📋 Copy Command"):
        st.code(command)
        st.success("Command ready to copy!")

with col2:
    if st.button("💾 Save Configuration"):
        config = {
            "model": {
                "classical_model": classical_model,
                "pretrained": pretrained,
                "freeze_layers": freeze_layers,
                "feature_dim": feature_dim,
                "n_qubits": n_qubits,
                "n_quantum_layers": n_quantum_layers,
            },
            "training": {
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
            }
        }

        config_path = project_root / "configs" / f"config_{run_name}.json"
        config_path.parent.mkdir(exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        st.success(f"Configuration saved to {config_path}")

with col3:
    start_training = st.button("🚀 Start Training", type="primary")

if start_training:
    st.session_state.training_active = True
    st.warning("""
    ⚠️ Training will start in terminal. Due to Streamlit limitations,
    please run the command above in your terminal to start training.

    Monitor progress in the terminal or check the "Experiments" page
    for MLflow tracking.
    """)

st.markdown("---")

# Training tips
with st.expander("💡 Training Tips & Best Practices"):
    st.markdown("""
    ### Hyperparameter Recommendations

    **For Fast Prototyping:**
    - Epochs: 5-10
    - Batch size: 128-256
    - Quantum layers: 2
    - Learning rate: 0.001

    **For Best Accuracy:**
    - Epochs: 20-50 (with early stopping)
    - Batch size: 64-128
    - Quantum layers: 3
    - Learning rate: 0.001 with scheduler

    **For Quantum Experiments:**
    - Try 2-6 qubits
    - Experiment with 1-4 quantum layers
    - Compare different quantum devices
    - Monitor quantum parameter evolution

    ### Common Issues

    - **Slow training**: Increase batch size, reduce quantum layers
    - **Overfitting**: Enable early stopping, reduce model capacity
    - **Unstable training**: Reduce learning rate, enable gradient clipping
    - **Low accuracy**: Increase epochs, try different learning rates
    """)

# Recent experiments
st.markdown("### 📈 Recent Experiments")

try:
    models_dir = project_root / "models" / "checkpoints"
    if models_dir.exists():
        checkpoints = list(models_dir.glob("*.pt"))
        if checkpoints:
            st.info(f"Found {len(checkpoints)} saved checkpoints")

            for ckpt in checkpoints[:5]:
                st.text(f"📁 {ckpt.name}")
        else:
            st.info("No checkpoints found. Train a model to get started!")
    else:
        st.info("No models directory yet. Start training to create checkpoints!")
except Exception as e:
    st.warning(f"Could not load checkpoints: {e}")
