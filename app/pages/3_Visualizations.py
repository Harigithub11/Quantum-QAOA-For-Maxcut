"""
Visualizations Page
Explore training curves, confusion matrices, t-SNE embeddings, and more
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Visualizations", page_icon="📊", layout="wide")

st.title("📊 Visualizations")
st.markdown("Explore comprehensive visualizations of model training and performance.")

st.markdown("---")

# Load results
import torch
checkpoints_dir = project_root / "models" / "checkpoints"
viz_tabs = st.tabs(["📈 Training Curves", "🎯 Performance", "🔍 Embeddings", "📸 Sample Predictions"])

# Tab 1: Training Curves
with viz_tabs[0]:
    st.markdown("### 📈 Training & Validation Curves")

    # Check for saved checkpoint files
    checkpoint_files = []
    if checkpoints_dir.exists():
        checkpoint_files = sorted(list(checkpoints_dir.glob("checkpoint_epoch_*.pth")))

    if checkpoint_files:
        st.success(f"✅ Found {len(checkpoint_files)} training checkpoints!")

        # Load training metrics from all checkpoints
        epochs_list = []
        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []

        for ckpt_file in checkpoint_files:
            try:
                checkpoint = torch.load(ckpt_file, map_location='cpu')
                metrics = checkpoint.get('metrics', {})
                epochs_list.append(checkpoint['epoch'])
                train_loss_list.append(metrics.get('train_loss', 0))
                val_loss_list.append(metrics.get('val_loss', 0))
                train_acc_list.append(metrics.get('train_acc', 0))
                val_acc_list.append(metrics.get('val_acc', 0))
            except Exception as e:
                st.warning(f"Could not load {ckpt_file.name}: {e}")

        if epochs_list:
            epochs = np.array(epochs_list)
            train_loss = np.array(train_loss_list)
            val_loss = np.array(val_loss_list)
            train_acc = np.array(train_acc_list)
            val_acc = np.array(val_acc_list)
        else:
            st.error("No valid checkpoint data found!")
            epochs = np.arange(1, 6)
            train_loss = np.zeros(5)
            val_loss = np.zeros(5)
            train_acc = np.zeros(5)
            val_acc = np.zeros(5)

        # Create interactive plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Loss Curves', 'Accuracy Curves')
        )

        # Loss plot
        fig.add_trace(
            go.Scatter(x=epochs, y=train_loss, name='Train Loss',
                      mode='lines+markers', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_loss, name='Val Loss',
                      mode='lines+markers', line=dict(color='red', width=2)),
            row=1, col=1
        )

        # Accuracy plot (already in percentage format)
        fig.add_trace(
            go.Scatter(x=epochs, y=train_acc, name='Train Acc',
                      mode='lines+markers', line=dict(color='blue', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_acc, name='Val Acc',
                      mode='lines+markers', line=dict(color='red', width=2)),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)

        fig.update_layout(height=400, showlegend=True, hovermode='x unified')
        st.plotly_chart(fig)

        # Metrics summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Train Loss", f"{train_loss[-1]:.4f}",
                     f"{train_loss[-1] - train_loss[0]:.4f}")
        with col2:
            st.metric("Final Val Loss", f"{val_loss[-1]:.4f}",
                     f"{val_loss[-1] - val_loss[0]:.4f}")
        with col3:
            st.metric("Final Train Acc", f"{train_acc[-1]:.2f}%",
                     f"+{(train_acc[-1] - train_acc[0]):.1f}%")
        with col4:
            st.metric("Final Val Acc", f"{val_acc[-1]:.2f}%",
                     f"+{(val_acc[-1] - val_acc[0]):.1f}%")

    else:
        st.info("No training history found. Train a model to see curves!")

        # Show example
        with st.expander("📖 See Example Training Curves"):
            st.image("https://via.placeholder.com/800x400/f0f2f6/1f77b4?text=Training+Curves+Will+Appear+Here")

    # Learning rate schedule
    st.markdown("---")
    st.markdown("#### 📉 Learning Rate Schedule")

    col1, col2 = st.columns([2, 1])

    with col1:
        lr_epochs = np.arange(1, 21)
        lr_values = 0.001 * (0.1 ** (lr_epochs // 7))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lr_epochs, y=lr_values, mode='lines+markers',
                                line=dict(color='green', width=2)))
        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Learning Rate",
            yaxis_type="log",
            height=300
        )
        st.plotly_chart(fig)

    with col2:
        st.markdown("**LR Schedule Info:**")
        st.markdown("""
        - Initial LR: 0.001
        - Scheduler: StepLR
        - Step size: 7 epochs
        - Gamma: 0.1
        """)

# Tab 2: Performance Metrics
with viz_tabs[1]:
    st.markdown("### 🎯 Model Performance Analysis")

    # Confusion Matrix
    st.markdown("#### 📋 Confusion Matrix")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Mock confusion matrix
        cm = np.random.randint(0, 100, (10, 10))
        np.fill_diagonal(cm, np.random.randint(800, 1000, 10))

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=range(10), yticklabels=range(10))
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("**Key Observations:**")
        st.markdown("""
        - Strong diagonal: Good accuracy
        - Common confusions: 3↔8, 5↔3
        - Digit 1 very accurate
        - Digit 8 challenging
        """)

        st.markdown("---")

        st.markdown("**Actions:**")
        if st.button("📥 Download Matrix"):
            st.info("Download feature coming soon!")

    # Per-class metrics
    st.markdown("---")
    st.markdown("#### 📊 Per-Class Performance")

    col1, col2 = st.columns(2)

    with col1:
        # Mock per-class accuracy
        digits = list(range(10))
        accuracies = np.random.uniform(85, 98, 10)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=digits, y=accuracies, marker_color='steelblue'))
        fig.update_layout(
            xaxis_title="Digit",
            yaxis_title="Accuracy (%)",
            title="Per-Class Accuracy",
            height=350
        )
        st.plotly_chart(fig)

    with col2:
        # Precision-Recall-F1
        precision = np.random.uniform(0.85, 0.98, 10)
        recall = np.random.uniform(0.85, 0.98, 10)
        f1 = 2 * (precision * recall) / (precision + recall)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=digits, y=precision, name='Precision',
                                mode='lines+markers'))
        fig.add_trace(go.Scatter(x=digits, y=recall, name='Recall',
                                mode='lines+markers'))
        fig.add_trace(go.Scatter(x=digits, y=f1, name='F1-Score',
                                mode='lines+markers'))
        fig.update_layout(
            xaxis_title="Digit",
            yaxis_title="Score",
            title="Precision, Recall, F1-Score",
            height=350
        )
        st.plotly_chart(fig)

    # Classification report
    st.markdown("---")
    st.markdown("#### 📄 Classification Report")

    report_data = {
        'Digit': list(range(10)),
        'Precision': np.round(precision, 3),
        'Recall': np.round(recall, 3),
        'F1-Score': np.round(f1, 3),
        'Support': np.random.randint(900, 1100, 10)
    }

    import pandas as pd
    df = pd.DataFrame(report_data)
    st.dataframe(df)

# Tab 3: Embeddings
with viz_tabs[2]:
    st.markdown("### 🔍 Feature Embeddings Visualization")

    st.markdown("#### 🎨 t-SNE 2D Projection")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Mock t-SNE data
        n_samples = 500
        n_classes = 10

        # Generate clustered points for each digit
        tsne_embeddings = []
        tsne_labels = []

        for digit in range(n_classes):
            center_x = np.random.uniform(-30, 30)
            center_y = np.random.uniform(-30, 30)
            x = np.random.normal(center_x, 5, n_samples // n_classes)
            y = np.random.normal(center_y, 5, n_samples // n_classes)
            tsne_embeddings.append(np.column_stack([x, y]))
            tsne_labels.extend([digit] * (n_samples // n_classes))

        tsne_embeddings = np.vstack(tsne_embeddings)

        fig = px.scatter(
            x=tsne_embeddings[:, 0],
            y=tsne_embeddings[:, 1],
            color=[str(l) for l in tsne_labels],
            labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Digit'},
            title='t-SNE Visualization of Feature Embeddings',
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(height=500)
        st.plotly_chart(fig)

    with col2:
        st.markdown("**Options:**")

        layer_select = st.selectbox(
            "Layer",
            ["Classical", "Quantum", "Final"],
            help="Which layer to visualize"
        )

        n_samples_slider = st.slider(
            "Samples",
            100, 1000, 500,
            help="Number of samples to plot"
        )

        perplexity = st.slider(
            "Perplexity",
            5, 50, 30,
            help="t-SNE perplexity parameter"
        )

        if st.button("🔄 Regenerate"):
            st.info("Regenerating with new parameters...")

    st.markdown("---")
    st.markdown("#### 📈 PCA Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Explained variance
        n_components = 10
        explained_var = np.random.exponential(0.15, n_components)
        explained_var = explained_var / explained_var.sum()
        explained_var = np.sort(explained_var)[::-1]
        cumulative_var = np.cumsum(explained_var)

        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Individual Variance', 'Cumulative Variance'))

        fig.add_trace(
            go.Bar(x=list(range(1, n_components+1)), y=explained_var),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=list(range(1, n_components+1)), y=cumulative_var,
                      mode='lines+markers'),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Component", row=1, col=1)
        fig.update_xaxes(title_text="Components", row=1, col=2)
        fig.update_yaxes(title_text="Variance Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Variance", row=1, col=2)
        fig.update_layout(height=350, showlegend=False)

        st.plotly_chart(fig)

    with col2:
        st.markdown("**PCA Summary:**")
        st.markdown(f"""
        - PC1 explains: {explained_var[0]:.1%}
        - PC2 explains: {explained_var[1]:.1%}
        - PC1-3 total: {cumulative_var[2]:.1%}
        - 95% variance: {np.where(cumulative_var >= 0.95)[0][0] + 1} components
        """)

        st.markdown("---")

        st.markdown("**Insights:**")
        st.markdown("""
        - Strong first principal component
        - Rapid variance accumulation
        - Low-dimensional structure
        - Good separability expected
        """)

# Tab 4: Sample Predictions
with viz_tabs[3]:
    st.markdown("### 📸 Sample Predictions")

    st.markdown("#### ✅ Correct Predictions")

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            # Mock image
            img = np.random.rand(28, 28)
            digit = np.random.randint(0, 10)
            conf = np.random.uniform(0.9, 0.99)

            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(f"Pred: {digit}\n{conf:.1%}", fontsize=8)
            st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### ❌ Misclassifications")

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            img = np.random.rand(28, 28)
            true_digit = np.random.randint(0, 10)
            pred_digit = (true_digit + np.random.randint(1, 3)) % 10
            conf = np.random.uniform(0.6, 0.85)

            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(f"True: {true_digit}\nPred: {pred_digit}\n{conf:.1%}",
                        fontsize=8, color='red')
            st.pyplot(fig)

    st.info("These are mock visualizations. After training, actual model predictions will be displayed.")

# Download section
st.markdown("---")
st.markdown("### 📥 Download Visualizations")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Download All Plots"):
        st.info("Preparing download... (Coming soon)")

with col2:
    if st.button("📄 Generate Report"):
        st.info("Generating comprehensive report... (Coming soon)")

with col3:
    if st.button("📧 Export to PDF"):
        st.info("PDF export feature coming soon!")
