"""
Experiments Page
Compare multiple training runs and explore hyperparameter impact
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Experiments", page_icon="🔬", layout="wide")

st.title("🔬 Experiment Tracking & Comparison")
st.markdown("Compare multiple training runs and analyze hyperparameter impact with MLflow.")

st.markdown("---")

# MLflow connection status
st.markdown("### 📊 MLflow Integration")

col1, col2, col3 = st.columns(3)

with col1:
    mlflow_status = st.selectbox(
        "MLflow Status",
        ["Connected ✓", "Disconnected ✗", "Local Mode"],
        index=2
    )

with col2:
    experiment_filter = st.selectbox(
        "Filter Experiments",
        ["All Experiments", "Quantum Models", "Classical Baseline", "Hybrid Models"],
        index=3
    )

with col3:
    if st.button("🔄 Refresh Experiments"):
        st.cache_data.clear()
        st.success("Refreshed!")

st.markdown("---")

# Mock experiment data
experiments_data = {
    'Run ID': [f'run_{i:03d}' for i in range(1, 11)],
    'Name': [f'hybrid_q4l{i%3+1}' for i in range(10)],
    'Test Accuracy': np.random.uniform(0.75, 0.95, 10),
    'Val Accuracy': np.random.uniform(0.72, 0.93, 10),
    'Train Loss': np.random.uniform(0.1, 0.5, 10),
    'Epochs': np.random.randint(5, 20, 10),
    'Batch Size': np.random.choice([32, 64, 128, 256], 10),
    'Learning Rate': np.random.choice([0.0001, 0.0005, 0.001], 10),
    'Quantum Layers': np.random.randint(1, 4, 10),
    'Duration (min)': np.random.uniform(10, 60, 10)
}

df_experiments = pd.DataFrame(experiments_data)
df_experiments['Test Accuracy'] = df_experiments['Test Accuracy'].round(4)
df_experiments['Val Accuracy'] = df_experiments['Val Accuracy'].round(4)
df_experiments['Train Loss'] = df_experiments['Train Loss'].round(4)
df_experiments['Duration (min)'] = df_experiments['Duration (min)'].round(1)

# Experiments table
st.markdown("### 📋 All Experiments")

st.dataframe(
    df_experiments,
    column_config={
        "Test Accuracy": st.column_config.ProgressColumn(
            "Test Accuracy",
            format="%.2%%",
            min_value=0,
            max_value=1,
        ),
    }
)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Download CSV"):
        st.info("CSV download feature coming soon!")

with col2:
    if st.button("📊 Generate Report"):
        st.info("Report generation coming soon!")

with col3:
    selected_runs = st.multiselect(
        "Select runs to compare",
        df_experiments['Run ID'].tolist(),
        max_selections=5
    )

st.markdown("---")

# Comparison visualizations
if selected_runs:
    st.markdown("### 📊 Run Comparison")

    df_selected = df_experiments[df_experiments['Run ID'].isin(selected_runs)]

    col1, col2 = st.columns(2)

    with col1:
        # Accuracy comparison
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df_selected['Run ID'],
            y=df_selected['Test Accuracy'],
            name='Test Accuracy',
            marker_color='steelblue'
        ))

        fig.add_trace(go.Bar(
            x=df_selected['Run ID'],
            y=df_selected['Val Accuracy'],
            name='Val Accuracy',
            marker_color='orange'
        ))

        fig.update_layout(
            title="Accuracy Comparison",
            yaxis_title="Accuracy",
            barmode='group',
            height=350
        )

        st.plotly_chart(fig)

    with col2:
        # Training duration vs accuracy
        fig = px.scatter(
            df_selected,
            x='Duration (min)',
            y='Test Accuracy',
            size='Epochs',
            color='Quantum Layers',
            hover_data=['Run ID', 'Batch Size', 'Learning Rate'],
            title="Duration vs Accuracy"
        )

        fig.update_layout(height=350)
        st.plotly_chart(fig)

st.markdown("---")

# Hyperparameter analysis
st.markdown("### 🔧 Hyperparameter Impact Analysis")

tab1, tab2, tab3, tab4 = st.tabs([
    "Learning Rate",
    "Batch Size",
    "Quantum Layers",
    "Training Duration"
])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        # Learning rate vs accuracy
        fig = px.box(
            df_experiments,
            x='Learning Rate',
            y='Test Accuracy',
            title="Learning Rate Impact on Test Accuracy"
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig)

    with col2:
        st.markdown("**Analysis:**")

        for lr in df_experiments['Learning Rate'].unique():
            mask = df_experiments['Learning Rate'] == lr
            avg_acc = df_experiments[mask]['Test Accuracy'].mean()
            st.markdown(f"- **LR {lr}**: {avg_acc:.2%} avg accuracy")

        st.markdown("---")

        st.markdown("**Recommendation:**")
        best_lr = df_experiments.groupby('Learning Rate')['Test Accuracy'].mean().idxmax()
        st.success(f"Best: {best_lr}")

with tab2:
    col1, col2 = st.columns([2, 1])

    with col1:
        # Batch size comparison
        fig = px.violin(
            df_experiments,
            x='Batch Size',
            y='Test Accuracy',
            title="Batch Size Impact on Accuracy",
            box=True
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig)

    with col2:
        st.markdown("**Trade-offs:**")

        st.markdown("""
        **Small batches (32-64):**
        - ✓ More updates
        - ✓ Better generalization
        - ✗ Slower training
        - ✗ Noisy gradients

        **Large batches (128-256):**
        - ✓ Faster training
        - ✓ Stable gradients
        - ✗ Less updates
        - ✗ May overfit
        """)

with tab3:
    col1, col2 = st.columns([2, 1])

    with col1:
        # Quantum layers impact
        fig = px.scatter(
            df_experiments,
            x='Quantum Layers',
            y='Test Accuracy',
            size='Duration (min)',
            color='Learning Rate',
            title="Quantum Layers vs Accuracy"
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig)

    with col2:
        st.markdown("**Observations:**")

        for ql in sorted(df_experiments['Quantum Layers'].unique()):
            mask = df_experiments['Quantum Layers'] == ql
            avg_acc = df_experiments[mask]['Test Accuracy'].mean()
            avg_time = df_experiments[mask]['Duration (min)'].mean()
            st.markdown(f"**{ql} layers:**")
            st.markdown(f"  - Accuracy: {avg_acc:.2%}")
            st.markdown(f"  - Time: {avg_time:.1f} min")

        st.markdown("---")

        st.info("Sweet spot: 2-3 quantum layers")

with tab4:
    col1, col2 = st.columns([2, 1])

    with col1:
        # Training efficiency
        fig = px.scatter(
            df_experiments,
            x='Epochs',
            y='Test Accuracy',
            size='Duration (min)',
            color='Batch Size',
            title="Training Epochs vs Accuracy",
            hover_data=['Run ID']
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig)

    with col2:
        st.markdown("**Efficiency Metrics:**")

        # Best runs
        best_3 = df_experiments.nlargest(3, 'Test Accuracy')

        st.markdown("**Top 3 Runs:**")
        for idx, row in best_3.iterrows():
            st.markdown(f"{row['Run ID']}: {row['Test Accuracy']:.2%}")

        st.markdown("---")

        fastest = df_experiments.loc[df_experiments['Duration (min)'].idxmin()]
        st.markdown(f"**Fastest Run:** {fastest['Run ID']}")
        st.markdown(f"Time: {fastest['Duration (min)']:.1f} min")

st.markdown("---")

# Best configuration
st.markdown("### 🏆 Best Configuration")

best_run = df_experiments.loc[df_experiments['Test Accuracy'].idxmax()]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📊 Performance")
    st.metric("Test Accuracy", f"{best_run['Test Accuracy']:.2%}")
    st.metric("Val Accuracy", f"{best_run['Val Accuracy']:.2%}")
    st.metric("Train Loss", f"{best_run['Train Loss']:.4f}")

with col2:
    st.markdown("#### ⚙️ Hyperparameters")
    st.metric("Learning Rate", best_run['Learning Rate'])
    st.metric("Batch Size", best_run['Batch Size'])
    st.metric("Quantum Layers", best_run['Quantum Layers'])

with col3:
    st.markdown("#### ⏱️ Training Info")
    st.metric("Epochs", best_run['Epochs'])
    st.metric("Duration", f"{best_run['Duration (min)']:.1f} min")
    st.metric("Run ID", best_run['Run ID'])

st.markdown("---")

if st.button("📋 Copy Best Configuration"):
    config_cmd = f"""
python train.py \\
  --batch-size {int(best_run['Batch Size'])} \\
  --epochs {int(best_run['Epochs'])} \\
  --learning-rate {best_run['Learning Rate']} \\
  --quantum-layers {int(best_run['Quantum Layers'])}
"""
    st.code(config_cmd, language="bash")
    st.success("Configuration ready to copy!")

# Parallel coordinates plot
st.markdown("---")
st.markdown("### 🎯 Hyperparameter Relationships")

fig = px.parallel_coordinates(
    df_experiments,
    dimensions=['Learning Rate', 'Batch Size', 'Quantum Layers', 'Epochs', 'Test Accuracy'],
    color='Test Accuracy',
    color_continuous_scale=px.colors.sequential.Viridis,
    title="Parallel Coordinates: Hyperparameters vs Accuracy"
)

fig.update_layout(height=500)
st.plotly_chart(fig)

st.info("""
💡 **How to read:** Each line represents one experiment. Lines colored in yellow/green
have higher test accuracy. Follow lines to see which hyperparameter combinations work best.
""")

st.markdown("---")

# Statistics summary
st.markdown("### 📈 Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Runs",
        len(df_experiments),
        "All time"
    )

with col2:
    st.metric(
        "Best Accuracy",
        f"{df_experiments['Test Accuracy'].max():.2%}",
        f"+{(df_experiments['Test Accuracy'].max() - df_experiments['Test Accuracy'].min()) * 100:.1f}%"
    )

with col3:
    st.metric(
        "Avg Duration",
        f"{df_experiments['Duration (min)'].mean():.1f} min",
        f"Total: {df_experiments['Duration (min)'].sum():.0f} min"
    )

with col4:
    st.metric(
        "Experiments >90%",
        len(df_experiments[df_experiments['Test Accuracy'] > 0.9]),
        f"{len(df_experiments[df_experiments['Test Accuracy'] > 0.9]) / len(df_experiments) * 100:.0f}%"
    )

# Tips
with st.expander("💡 Experiment Tracking Best Practices"):
    st.markdown("""
    ### Effective Experiment Management

    **Organization:**
    - Use descriptive experiment names
    - Tag runs with metadata (quantum vs classical, etc.)
    - Document key decisions and hypotheses
    - Archive unsuccessful experiments for learning

    **Comparison Strategy:**
    - Change one hyperparameter at a time
    - Run multiple seeds for reliability
    - Track computational cost alongside accuracy
    - Compare against established baselines

    **MLflow Features:**
    - Automatic metric logging
    - Model versioning and registration
    - Artifact storage (models, plots)
    - Collaborative experiment sharing

    **Analysis Tips:**
    - Look for trends, not single best runs
    - Consider trade-offs (accuracy vs speed)
    - Validate on multiple datasets if possible
    - Document unexpected findings
    """)
