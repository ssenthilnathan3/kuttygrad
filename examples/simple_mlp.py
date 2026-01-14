"""
Simple MLP training example
"""
import numpy as np
from kuttygrad import Tensor
from kuttygrad.nn import Linear, ReLU, Sequential


def main():
    np.random.seed(42)
    
    # Create a simple 2-layer MLP
    model = Sequential(
        Linear(10, 32, bias=True),
        ReLU(),
        Linear(32, 1, bias=True)
    )
    
    # Create dummy data
    batch_size = 64
    x = Tensor(np.random.randn(batch_size, 10))
    y = Tensor(np.random.randn(batch_size, 1))
    
    # Hyperparameters
    learning_rate = 0.01
    epochs = 50
    
    print("Training a simple MLP...")
    print(f"Model structure: 10 -> 32 -> 1")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}\n")
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        pred = model(x)
        
        # Mean squared error loss
        loss = ((pred - y) ** 2).sum() / float(batch_size)
        
        # Backward pass
        loss.backward()
        
        # Manual gradient descent update
        for param in model.parameters():
            param.data -= learning_rate * param.grad.data
            param.grad = None  # Reset gradients for next iteration
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} | Loss: {loss.data:.6f}")
    
    print("\nTraining completed!")
    
    # Test on new data
    x_test = Tensor(np.random.randn(10, 10))
    y_pred = model(x_test)
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Sample predictions: {y_pred.data[:3].flatten()}")


if __name__ == "__main__":
    main()
