"""
XOR learning example - teaching a neural network to learn XOR function
"""
import numpy as np
from kuttygrad import Tensor
from kuttygrad.nn import Linear, ReLU, Sequential


def main():
    np.random.seed(42)
    
    # XOR truth table
    x_data = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    
    y_data = np.array([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ])
    
    x = Tensor(x_data)
    y = Tensor(y_data)
    
    # Create model
    model = Sequential(
        Linear(2, 4, bias=True),
        ReLU(),
        Linear(4, 1, bias=True)
    )
    
    # Hyperparameters
    learning_rate = 0.1
    epochs = 500
    
    print("Training a neural network to learn XOR...")
    print(f"Model structure: 2 -> 4 -> 1")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}\n")
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        pred = model(x)
        
        # Mean squared error loss
        loss = ((pred - y) ** 2).sum()
        losses.append(loss.data.copy())
        
        # Backward pass
        loss.backward()
        
        # Manual gradient descent update
        for param in model.parameters():
            param.data -= learning_rate * param.grad.data
            param.grad = None
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1:4d} | Loss: {loss.data:.6f}")
    
    print("\nTraining completed!")
    print("\nFinal predictions:")
    print("Input | Target | Prediction")
    print("------|--------|----------")
    
    final_pred = model(x)
    for i in range(len(x_data)):
        target = y_data[i, 0]
        pred = final_pred.data[i, 0]
        print(f"{x_data[i]} |  {target:.1f}   |   {pred:.4f}")


if __name__ == "__main__":
    main()
