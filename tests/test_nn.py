import numpy as np
import pytest
from kuttygrad import Tensor
from kuttygrad.nn import Linear, ReLU, Sigmoid, Tanh, Flatten, Sequential


class TestLinear:
    def test_linear_forward(self):
        linear = Linear(in_features=3, out_features=2, bias=True)
        x = Tensor(np.random.randn(4, 3))
        y = linear(x)
        assert y.shape == (4, 2)

    def test_linear_without_bias(self):
        linear = Linear(in_features=3, out_features=2, bias=False)
        x = Tensor(np.random.randn(4, 3))
        y = linear(x)
        assert y.shape == (4, 2)
        assert linear.bias is None

    def test_linear_parameters(self):
        linear = Linear(in_features=3, out_features=2, bias=True)
        params = list(linear.parameters())
        assert len(params) == 2  # weight and bias

    def test_linear_gradient(self):
        linear = Linear(in_features=3, out_features=1, bias=True)
        x = Tensor(np.ones((2, 3)))
        y = linear(x)
        loss = y.sum()
        loss.backward()
        
        # Check that gradients are computed
        assert linear.weight.grad is not None
        assert linear.bias.grad is not None


class TestReLU:
    def test_relu_forward(self):
        relu = ReLU()
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = relu(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert np.allclose(y.data, expected)

    def test_relu_gradient(self):
        x = Tensor([-1.0, 1.0], requires_grad=True)
        relu = ReLU()
        y = relu(x)
        y.sum().backward()
        # Gradient is 0 for negative, 1 for positive
        assert np.allclose(x.grad.data, np.array([0.0, 1.0]))


class TestSigmoid:
    def test_sigmoid_forward(self):
        sigmoid = Sigmoid()
        x = Tensor([0.0])
        y = sigmoid(x)
        assert np.isclose(y.data, 0.5)


class TestTanh:
    def test_tanh_forward(self):
        tanh = Tanh()
        x = Tensor([0.0])
        y = tanh(x)
        assert np.isclose(y.data, 0.0)


class TestFlatten:
    def test_flatten_forward(self):
        flatten = Flatten()
        x = Tensor(np.arange(24).reshape(2, 3, 4))
        y = flatten(x)
        assert y.shape == (2, 12)

    def test_flatten_preserves_batch(self):
        flatten = Flatten()
        x = Tensor(np.random.randn(5, 10, 20))
        y = flatten(x)
        assert y.shape == (5, 200)


class TestSequential:
    def test_sequential_forward(self):
        model = Sequential(
            Linear(10, 20, bias=True),
            ReLU(),
            Linear(20, 1, bias=True)
        )
        x = Tensor(np.random.randn(5, 10))
        y = model(x)
        assert y.shape == (5, 1)

    def test_sequential_parameters(self):
        model = Sequential(
            Linear(10, 20, bias=True),
            ReLU(),
            Linear(20, 1, bias=True)
        )
        params = list(model.parameters())
        # 2 linear layers * 2 params (weight + bias) = 4
        assert len(params) == 4

    def test_sequential_gradient(self):
        model = Sequential(
            Linear(10, 5, bias=True),
            ReLU(),
            Linear(5, 1, bias=True)
        )
        x = Tensor(np.random.randn(3, 10))
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Check that all parameters have gradients
        for param in model.parameters():
            assert param.grad is not None


class TestModule:
    def test_train_eval(self):
        model = Sequential(
            Linear(10, 20, bias=True),
            ReLU(),
            Linear(20, 1, bias=True)
        )
        
        model.train()
        assert model.training == True
        
        model.eval()
        assert model.training == False

    def test_modules_iterator(self):
        model = Sequential(
            Linear(10, 20, bias=True),
            ReLU(),
            Linear(20, 1, bias=True)
        )
        
        modules = list(model.modules())
        # Should have 2 Linear modules and 1 ReLU module
        assert len(modules) == 3


class TestEndToEnd:
    def test_simple_training(self):
        """Test a simple training loop"""
        np.random.seed(42)
        
        # Create model
        model = Sequential(
            Linear(5, 10, bias=True),
            ReLU(),
            Linear(10, 1, bias=True)
        )
        
        # Create dummy data
        x = Tensor(np.random.randn(10, 5))
        y = Tensor(np.random.randn(10, 1))
        
        # Initial loss
        pred = model(x)
        initial_loss = ((pred - y) ** 2).sum()
        
        # Training step
        initial_loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            # Update parameters
            param.data -= 0.01 * param.grad.data

    def test_mlp_classification(self):
        """Test a simple MLP that can learn XOR"""
        np.random.seed(42)
        
        # Create model
        model = Sequential(
            Linear(2, 4, bias=True),
            ReLU(),
            Linear(4, 1, bias=True)
        )
        
        # XOR data
        x = Tensor(np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]))
        y = Tensor(np.array([[0.0], [1.0], [1.0], [0.0]]))
        
        # Training loop
        learning_rate = 0.1
        initial_loss = None
        
        for epoch in range(100):
            # Forward pass
            pred = model(x)
            
            # Loss
            loss = ((pred - y) ** 2).sum()
            
            if epoch == 0:
                initial_loss = loss.data.copy()
            
            # Backward pass
            loss.backward()
            
            # Manual update
            for param in model.parameters():
                param.data -= learning_rate * param.grad.data
                param.grad = None
        
        # Final loss should be lower than initial
        final_loss = model(x)
        final_loss = ((final_loss - y) ** 2).sum()
        assert final_loss.data < initial_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
