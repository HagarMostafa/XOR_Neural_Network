# 🧠 Simple Neural Network from Scratch

A basic neural network implementation using only NumPy to solve the XOR problem. Built for learning and understanding neural network fundamentals.

## 📋 What This Does

This neural network learns the XOR function:
- Input: Two numbers (0 or 1)
- Output: XOR result (0 or 1)
- Uses backpropagation to learn from examples

## 🏗️ Network Structure

```
Input (2) → Hidden (8) → Output (1)
   X1 ────────── H1-H8 ────────── Y
   X2 ────────── [sigmoid] ──────
```

## 🚀 How to Run

### Requirements
```bash
pip install numpy matplotlib
```

### Run the Code
```python
# Just run the script
python neural_network.py
```

### For Jupyter Notebook
Copy the code into a Jupyter notebook cell and run it.

## 📊 Expected Results

After training, the network should output:

| Input | Expected | Network Output |
|-------|----------|----------------|
| [0,0] | 0        | ~0.02          |
| [0,1] | 1        | ~0.98          |
| [1,0] | 1        | ~0.98          |
| [1,1] | 0        | ~0.02          |

**Final Loss**: Should be less than 0.01  
**Training Time**: ~2000-5000 epochs

## 🔧 Key Components

### 1. **Network Class**
```python
SimpleNeuralNetwork(input_size=2, hidden_size=8, output_size=1)
```

### 2. **Training Process**
- Forward pass: Calculate predictions
- Calculate error (loss)
- Backward pass: Update weights
- Repeat until loss is low

### 3. **Important Functions**
- `forward()` - Makes predictions
- `backward()` - Updates weights using backpropagation
- `train()` - Runs the training loop
- `predict()` - Makes new predictions

## 📈 What You'll See

When you run the code:
```
Training Neural Network on XOR Problem
========================================
Training Data:
Input: [0 0] -> Output: 0
Input: [0 1] -> Output: 1
Input: [1 0] -> Output: 1
Input: [1 1] -> Output: 0

Epoch 0, Loss: 0.234567
Epoch 200, Loss: 0.156432
Epoch 400, Loss: 0.089234
...
Epoch 2000, Loss: 0.003456

Testing trained network:
==============================
Input: [0 0] -> Predicted: 0.0234, Actual: 0
Input: [0 1] -> Predicted: 0.9876, Actual: 1
Input: [1 0] -> Predicted: 0.9823, Actual: 1
Input: [1 1] -> Predicted: 0.0187, Actual: 0

Final loss: 0.003456
Network successfully learned XOR function!
```

Plus a graph showing loss decreasing over time.

## 🎓 What You'll Learn

- How neural networks make predictions (forward pass)
- How they learn from mistakes (backpropagation)
- Why proper weight initialization matters (Xavier initialization)
- How gradient descent works
- Why XOR needs a hidden layer

## ⚙️ Code Structure

```python
class SimpleNeuralNetwork:
    def __init__():          # Initialize weights and biases
    def sigmoid():           # Activation function
    def forward():           # Make predictions
    def backward():          # Calculate gradients
    def train():             # Training loop
    def predict():           # Make new predictions

# Training data (XOR problem)
create_xor_data()           # Creates the training examples

# Training and testing
nn = SimpleNeuralNetwork()  # Create network
nn.train(X, y)             # Train on XOR data
predictions = nn.predict(X) # Test the trained network
```

## 🔧 Customization Options

### Change Network Size
```python
# Smaller network
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Larger network  
nn = SimpleNeuralNetwork(input_size=2, hidden_size=16, output_size=1)
```

### Adjust Learning
```python
# Faster learning
nn = SimpleNeuralNetwork(learning_rate=1.0)

# Slower learning
nn = SimpleNeuralNetwork(learning_rate=0.1)

# More training
losses = nn.train(X, y, epochs=10000)
```


---

⭐ **If this helped you understand neural networks, please star the repo!** ⭐
