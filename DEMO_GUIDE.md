# 🎬 Demo Guide - Quantum-Classical ML for MNIST

This guide will walk you through a complete demonstration of the hybrid quantum-classical machine learning project.

## 📋 Prerequisites Check

Before starting the demo, ensure:
- [x] Python 3.9+ installed
- [x] All dependencies installed (`pip install -r requirements.txt`)
- [x] Gemini API key configured in `.env`
- [x] Training has completed (or use pre-trained model)

## 🚀 Complete Demo Workflow

### Step 1: Launch the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard will open at `http://localhost:8501`

**What you'll see:**
- Welcome page with project overview
- Architecture diagram
- System status metrics
- Navigation to 5 different pages

### Step 2: Explore Training Configuration (Page 1)

Navigate to "Train Model" page.

**Demo Actions:**
1. Review default hyperparameters:
   - Batch size: 64
   - Learning rate: 0.001
   - Quantum layers: 2
   - Epochs: 5

2. Adjust quantum parameters:
   - Try different qubit counts (2-8)
   - Change quantum layers (1-4)

3. Copy the training command or save configuration

**Key Point to Highlight:**
- Show how easy it is to configure both classical and quantum components
- Explain the trade-off between quantum layers and training speed

### Step 3: Make Predictions (Page 2)

Navigate to "Test Model" page.

**Demo Actions:**
1. Select a trained model checkpoint
2. Upload a handwritten digit image OR select from test set
3. Click "Run Prediction"
4. Review results:
   - Predicted digit
   - Confidence score
   - Probability distribution (bar chart)
   - Top-3 predictions

5. Enable "Use Gemini Explanations"
6. Get AI-powered explanation of the prediction

**Key Point to Highlight:**
- Real-time inference
- Gemini API provides natural language explanations
- Model confidence visualization

### Step 4: Explore Visualizations (Page 3)

Navigate to "Visualizations" page.

**Demo Actions:**
1. **Training Curves Tab:**
   - Show loss decreasing over epochs
   - Show accuracy increasing
   - Point out train/val gap (overfitting indicator)

2. **Performance Tab:**
   - Review confusion matrix
   - Identify which digits are confused (e.g., 3↔8, 5↔3)
   - Check per-class accuracy

3. **Embeddings Tab:**
   - Show t-SNE 2D projection
   - Explain how different digits cluster
   - Point out overlap regions (harder to classify)

4. **Sample Predictions Tab:**
   - Show correct predictions
   - Show misclassifications
   - Discuss why certain mistakes occur

**Key Points to Highlight:**
- Interactive Plotly visualizations
- t-SNE reveals learned feature structure
- Confusion matrix guides improvement efforts

### Step 5: Quantum Circuit Analysis (Page 4)

Navigate to "Quantum Analysis" page.

**Demo Actions:**
1. **Circuit Architecture:**
   - Show 4-qubit circuit diagram
   - Explain RY/RZ rotation gates
   - Explain CNOT entanglement

2. **Parameter Evolution:**
   - Show how quantum parameters change during training
   - Point out convergence patterns

3. **Quantum State Visualization:**
   - **Bloch Sphere:** Show individual qubit states
   - **State Vector:** Show measurement probability distribution
   - **Density Matrix:** Show quantum state properties

4. **Entanglement Analysis:**
   - Show entanglement entropy over training
   - Show pairwise qubit entanglement heatmap

5. **Quantum vs Classical Contribution:**
   - Show pie chart (e.g., 35% quantum, 65% classical)
   - Explain quantum advantage

**Key Points to Highlight:**
- Quantum circuit is trainable (parameters update via backprop)
- Entanglement helps capture feature correlations
- Small quantum contribution (8 parameters) provides meaningful improvement

### Step 6: Compare Experiments (Page 5)

Navigate to "Experiments" page.

**Demo Actions:**
1. Review all training runs in table
2. Select 2-3 runs to compare
3. View comparison charts:
   - Accuracy comparison (bar chart)
   - Duration vs accuracy (scatter)

4. Analyze hyperparameter impact:
   - Learning rate effect
   - Batch size trade-offs
   - Quantum layers impact

5. Check "Best Configuration" section
6. Copy the best training command

**Key Points to Highlight:**
- MLflow tracks all experiments automatically
- Easy to compare hyperparameters
- Parallel coordinates plot shows relationships

## 🎯 Key Demo Talking Points

### 1. Hybrid Architecture Benefits
- **Classical strength:** Robust feature extraction from images
- **Quantum strength:** Nonlinear transformations and entanglement
- **Together:** Achieve better accuracy with fewer parameters

### 2. Real-World Applicability
- Easily adaptable to other image datasets (CIFAR-10, Fashion-MNIST)
- Can scale quantum component for more complex tasks
- Ready for quantum hardware (just change PennyLane device)

### 3. Development Workflow
- Start with Streamlit UI for experimentation
- Use MLflow to track what works
- Deploy best model for production

### 4. AI-Powered Insights
- Gemini explains predictions in natural language
- Helps build trust in model decisions
- Useful for debugging misclassifications

## 🔧 Troubleshooting During Demo

### If training is slow:
- Explain: Quantum simulation is CPU-bound
- Solution: Use fewer quantum layers or batch size 256

### If Gemini API fails:
- Check API key in `.env`
- Fallback: Use mock explanations

### If visualizations don't load:
- Check results directory has training history files
- Fallback: Show example visualizations

## 📊 Demo Script (5-Minute Version)

**Minute 1:** Overview & Architecture
- Open main page, explain hybrid model
- Show architecture diagram

**Minute 2:** Training Configuration
- Navigate to Train Model page
- Show hyperparameter controls
- Explain quantum circuit configuration

**Minute 3:** Predictions & Explanations
- Navigate to Test Model page
- Make a prediction
- Show Gemini explanation

**Minute 4:** Visualizations
- Navigate to Visualizations page
- Show training curves
- Show confusion matrix
- Show t-SNE embeddings

**Minute 5:** Quantum Analysis
- Navigate to Quantum Analysis page
- Show circuit diagram
- Show parameter evolution
- Explain quantum advantage

## 📝 Demo Checklist

Before presenting:
- [ ] Streamlit app launches successfully
- [ ] At least one model is trained
- [ ] Gemini API key is configured
- [ ] Test image ready for prediction demo
- [ ] Browser zoomed to appropriate level for audience
- [ ] Backup slides prepared (in case of technical issues)

During demo:
- [ ] Start with overview slide
- [ ] Walk through each Streamlit page
- [ ] Highlight quantum circuit visualization
- [ ] Show prediction + Gemini explanation
- [ ] Compare multiple experiments
- [ ] End with future directions

After demo:
- [ ] Answer questions
- [ ] Provide GitHub repository link
- [ ] Share documentation

## 🎓 Advanced Demo Extensions

For longer presentations (15-20 minutes), add:

1. **Live Training:**
   - Start a training run from Streamlit
   - Show live progress in terminal
   - Explain why it's slow (quantum simulation)

2. **Code Walkthrough:**
   - Open `src/models/quantum.py`
   - Show quantum circuit implementation
   - Explain parameter-shift gradients

3. **Jupyter Notebook:**
   - Open `notebooks/02_model_development.ipynb`
   - Show step-by-step model building
   - Interactive quantum circuit execution

4. **Testing:**
   - Run `pytest tests/ -v`
   - Show test coverage
   - Explain importance of testing quantum code

5. **MLflow UI:**
   - Open MLflow tracking UI
   - Show detailed experiment tracking
   - Compare runs side-by-side

## 🌟 Closing Points

**Achievements:**
- ✅ Built hybrid quantum-classical model
- ✅ Achieved competitive accuracy on MNIST
- ✅ Created production-ready web application
- ✅ Comprehensive testing and documentation
- ✅ AI-powered explanations

**Future Directions:**
- Deploy on real quantum hardware (IBM, AWS Braket)
- Scale to more complex datasets (CIFAR-10, ImageNet)
- Experiment with different quantum ansätze
- Optimize quantum-classical interface

**Impact:**
- Demonstrates practical quantum machine learning
- Bridges gap between quantum and classical AI
- Provides template for future quantum ML projects

---

**Questions & Discussion**

Prepare to answer:
- Why hybrid instead of pure quantum?
- How do quantum gradients work?
- What about quantum noise?
- Can this run on real quantum hardware?
- What's the computational cost?

---

## 📞 Support

For demo questions or issues:
- GitHub Issues: [repository link]
- Documentation: README.md
- Contact: [your email/contact]

**Good luck with your demo! 🚀**
