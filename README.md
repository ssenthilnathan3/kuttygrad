# kuttygrad

A lightweight automatic differentiation framework for building and training neural networks in Python.

## Features

- **Tensor**: Core data structure with automatic gradient computation
- **Autograd**: Reverse-mode automatic differentiation (backpropagation)
- **Operations**: 20+ differentiable operations (add, mul, pow, matmul, sigmoid, relu, tanh, exp, log, etc.)
- **Neural Network Layers**: Linear, ReLU, Sigmoid, Tanh, Flatten, Sequential
- **Gradient Checking**: Built-in numerical gradient verification

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from kuttygrad import Tensor
from kuttygrad.nn import Linear, ReLU, Sequential
import numpy as np

# Create a simple 2-layer MLP
model = Sequential(
    Linear(784, 128, bias=True),
    ReLU(),
    Linear(128, 10, bias=True)
)

# Forward pass
x = Tensor(np.random.randn(32, 784))
output = model(x)

# Compute loss and backprop
loss = output.sum()
loss.backward()

# Access gradients
for param in model.parameters():
    print(f"Gradient shape: {param.grad.shape}")
```

## Core Components

### Tensor
The fundamental data structure representing arrays with gradient tracking:

```python
from kuttygrad import Tensor

# Create a tensor
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

# Operations
y = x + 1
z = y * 2

# Compute gradients
z.sum().backward()
print(x.grad)  # Tensor([2., 2., 2.])
```

### Operations
All operations are implemented with both forward and backward passes:

```python
x = Tensor([[1.0, 2.0], [3.0, 4.0]])

# Arithmetic
y = x + 1     # Add
z = x * 2     # Multiply
w = x ** 2    # Power

# Matrix operations
a = Tensor([[1.0, 2.0], [3.0, 4.0]])
b = Tensor([[5.0, 6.0], [7.0, 8.0]])
c = a @ b     # Matrix multiplication

# Activation functions
from kuttygrad.ops import Sigmoid, ReLU, TanH
sig = Sigmoid()(x)
relu = ReLU()(x)
tanh = TanH()(x)

# Other operations
from kuttygrad.ops import Log, Exp, Sqrt, Sum, Reshape, Transpose
```

### Neural Network Modules

```python
from kuttygrad.nn import Linear, ReLU, Flatten, Sequential

# Linear layer
fc = Linear(in_features=10, out_features=5, bias=True)

# Activation functions
relu = ReLU()
sigmoid = Sigmoid()
tanh = Tanh()

# Utility layers
flatten = Flatten()

# Sequential container
model = Sequential(
    Linear(784, 128, bias=True),
    ReLU(),
    Linear(128, 10, bias=True)
)
```

### Gradient Checking

Verify that gradients are computed correctly using numerical differentiation:

```python
from kuttygrad import gradcheck_fn

def f(x):
    return (x ** 2).sum()

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
gradcheck_fn(f, x)  # Raises AssertionError if check fails
```

## Training Example

```python
from kuttygrad import Tensor
from kuttygrad.nn import Linear, ReLU, Sequential
import numpy as np

# Create model
model = Sequential(
    Linear(10, 32, bias=True),
    ReLU(),
    Linear(32, 1, bias=True)
)

# Create dummy data
x = Tensor(np.random.randn(64, 10))
y = Tensor(np.random.randn(64, 1))

# Training loop
learning_rate = 0.01
for epoch in range(100):
    # Forward pass
    pred = model(x)
    
    # Loss
    loss = ((pred - y) ** 2).sum()
    
    # Backward pass
    loss.backward()
    
    # Manual update (gradient descent)
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data
        param.grad = None  # Reset gradients
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss.data.item():.4f}")
```

## Supported Operations

### Arithmetic
- Addition: `a + b`
- Subtraction: `a - b`
- Multiplication: `a * b`
- Division: `a / b`
- Power: `a ** b`
- Negation: `-a`

### Linear Algebra
- Matrix multiplication: `a @ b`
- Transpose: `a.transpose()`
- Reshape: `a.reshape(shape)`
- Sum: `a.sum(axes)`
- Broadcast: `a.broadcast_to(shape)`

### Activation Functions
- ReLU: `relu(x)`
- Sigmoid: `sigmoid(x)`
- Tanh: `tanh(x)`
- Exponential: `exp(x)`
- Natural log: `log(x)`
- Square root: `sqrt(x)`
- Absolute value: `abs(x)`

## Architecture

```
kuttygrad/
├── tensor.py          # Tensor class with autograd
├── function.py        # Base Function class for operations
├── gradcheck.py       # Gradient checking utility
├── ops/               # Operation implementations
│   ├── add.py
│   ├── mul.py
│   ├── pow.py
│   ├── matmul.py
│   ├── sigmoid.py
│   ├── relu.py
│   ├── tanh.py
│   ├── exp.py
│   ├── log.py
│   └── ...
└── nn/                # Neural network modules
    ├── module.py      # Base Module class
    ├── linear.py      # Linear layer
    ├── relu.py        # ReLU activation
    ├── sigmoid.py     # Sigmoid activation
    ├── tanh.py        # Tanh activation
    ├── flatten.py     # Flatten layer
    ├── sequential.py  # Sequential container
    └── functional.py  # Functional API
```

## Implementation Details

### Automatic Differentiation
kuttygrad uses reverse-mode automatic differentiation (backpropagation):

1. **Forward pass**: Operations build a computation graph
2. **Backward pass**: Gradients are computed in reverse topological order
3. **Chain rule**: Each operation implements both `forward()` and `backward()`

### Tensor Properties
- `data`: The underlying NumPy array
- `requires_grad`: Whether to track gradients (default: True)
- `grad`: Accumulated gradient (set after backward pass)
- `device`: CPU or GPU (currently CPU only)
- `dtype`: Data type (e.g., float32)

### Graph Construction
When `requires_grad=True`, tensors track:
- `_op`: The operation that created this tensor
- `_inputs`: Parent tensors in the computation graph

## License

MIT
