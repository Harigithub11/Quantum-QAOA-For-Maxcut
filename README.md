# Hybrid Neural Networks for MNIST Classification

A state-of-the-art implementation of a hybrid neural network that combines ResNet18 feature extraction with a 4-qubit variational quantum circuit for handwritten digit classification on the MNIST dataset.

## 🎯 Project Overview

This project demonstrates the practical integration of quantum computing with classical deep learning by:
- Using **ResNet18** (pre-trained on ImageNet) for classical feature extraction
- Processing features through a **4-qubit variational quantum circuit** using PennyLane
- Achieving competitive accuracy (~98-99%) on MNIST digit classification
- Providing an interactive **Streamlit web application** for training and testing
- Tracking experiments with **MLflow** for reproducibility
- Generating AI-powered explanations using **Gemini API**

### Architecture

```
MNIST Image (28×28)
    ↓
ResNet18 Feature Extractor (frozen, pre-trained)
    ↓
Feature Vector (4D)
    ↓
Quantum Circuit (4 qubits, 3 variational layers)
  - Angle Encoding
  - RY Rotation Gates
  - CNOT Entanglement
  - Pauli-Z Measurement
    ↓
Classical Output Layer (10 classes)
    ↓
Predicted Digit (0-9)
```

## 📁 Project Structure

```
quantum-ml-mnist/
├── data/                   # MNIST dataset (auto-downloaded)
├── models/                 # Saved model checkpoints
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── data/             # Data loading & preprocessing
│   ├── models/           # Model architectures
│   ├── training/         # Training & evaluation
│   ├── visualization/    # Plotting & visualization
│   └── gemini_integration/  # AI explanations
├── app/                   # Streamlit web application
├── configs/               # Configuration files
├── tests/                 # Unit tests
├── logs/                  # TensorBoard logs
├── mlruns/               # MLflow experiment tracking
└── results/              # Generated figures & reports
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional, for faster training)
- Gemini API key (for AI explanations)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd quantum-ml-mnist
```

2. **Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your_api_key_here
```

### Running the Project

#### Option 1: Streamlit Web Application (Recommended)

```bash
streamlit run app/streamlit_app.py
```

This will open an interactive web interface where you can:
- Configure and train the model
- Test on custom images
- View visualizations and metrics
- Compare experiments
- Get AI-powered explanations

#### Option 2: Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to `notebooks/` and explore:
- `01_data_exploration.ipynb` - Explore the MNIST dataset
- `02_model_development.ipynb` - Build and test the model
- `03_results_analysis.ipynb` - Analyze training results

#### Option 3: Command Line Training

```bash
python -m src.training.train \
    --config configs/training_config.yaml \
    --model-config configs/model_config.yaml
```

## 📊 Features

### 1. **Hybrid Quantum-Classical Model**
- Pre-trained ResNet18 for robust feature extraction
- 4-qubit variational quantum circuit with:
  - Angle encoding for classical-to-quantum data mapping
  - Trainable RY rotation gates
  - CNOT gates for quantum entanglement
  - Pauli-Z expectation value measurements

### 2. **Advanced Visualization**
- Real-time training progress (loss, accuracy curves)
- Interactive Plotly dashboards
- Confusion matrices and classification reports
- Quantum circuit diagrams
- TensorBoard integration

### 3. **Experiment Tracking**
- MLflow for comprehensive experiment management
- Model versioning and artifact logging
- Hyperparameter tracking
- Performance comparison across runs

### 4. **AI-Powered Features**
- Prediction explanations using Gemini API
- Automated report generation
- Insights about quantum vs. classical contributions

### 5. **Web Application**
- Interactive training interface
- Real-time inference on uploaded images
- Experiment comparison dashboard
- Export results and visualizations

## 🔧 Configuration

### Model Configuration (`configs/model_config.yaml`)

```yaml
classical:
  model_name: "resnet18"
  pretrained: true
  feature_dim: 4

quantum:
  n_qubits: 4
  n_layers: 3
  device: "default.qubit"
  encoding: "angle"
  entanglement: "linear"
```

### Training Configuration (`configs/training_config.yaml`)

```yaml
training:
  num_epochs: 50
  learning_rate: 0.001
  batch_size: 64

early_stopping:
  enabled: true
  patience: 10
```

## 📈 Results

### Expected Performance

- **Test Accuracy**: ~98-99% on MNIST
- **Training Time**: ~2-3x slower than pure classical (due to quantum simulation)
- **Quantum Contribution**: ~3-4% accuracy improvement from entanglement

### Visualization Examples

The project generates:
- Training/validation loss curves
- Accuracy progression plots
- Confusion matrices
- Quantum circuit diagrams
- Parameter evolution visualizations

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## 📚 Documentation

### Key Modules

- **`src/data/dataset.py`**: MNIST data loading and preprocessing
- **`src/models/quantum.py`**: 4-qubit variational quantum circuit
- **`src/models/classical.py`**: ResNet18 feature extractor
- **`src/models/hybrid.py`**: Full hybrid model integration
- **`src/training/train.py`**: Training loop with MLflow logging
- **`src/training/evaluate.py`**: Model evaluation and metrics
- **`src/visualization/plots.py`**: Visualization utilities
- **`src/gemini_integration/explainer.py`**: AI-powered explanations

### Configuration Files

- **`configs/model_config.yaml`**: Model architecture settings
- **`configs/training_config.yaml`**: Training hyperparameters
- **`.env`**: Environment variables (API keys, paths)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is part of the SRM University Quantum Machine Learning course (7th Semester).

## 🙏 Acknowledgements

### Frameworks & Tools
- **PennyLane** - Quantum machine learning framework by Xanadu AI
- **PyTorch** - Deep learning framework
- **Streamlit** - Web application framework
- **MLflow** - Experiment tracking
- **Google Gemini** - AI-powered explanations

### Academic Resources
- MNIST Dataset: LeCun et al. (1998)
- ResNet Architecture: He et al. (2016)
- Hybrid Quantum ML: Recent advances in quantum computing research

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact the project maintainers.

---

**Built with ❤️ for advancing quantum machine learning research**

**SRM University | Quantum Machine Learning Course | 7th Semester**
