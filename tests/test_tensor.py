import numpy as np
import pytest
from kuttygrad import Tensor, gradcheck_fn


class TestTensorBasics:
    def test_tensor_creation(self):
        # From list
        t = Tensor([1, 2, 3])
        assert t.shape == (3,)
        
        # From ndarray
        arr = np.array([[1, 2], [3, 4]])
        t = Tensor(arr)
        assert t.shape == (2, 2)
        
        # From scalar
        t = Tensor(5.0)
        assert t.shape == ()

    def test_tensor_properties(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        assert t.shape == (2, 3)
        assert t.ndim == 2
        assert t.size == 6
        assert t.requires_grad == True

    def test_detach(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = x.detach()
        assert y.requires_grad == False
        assert np.allclose(x.data, y.data)

    def test_numpy(self):
        x = Tensor([1, 2, 3])
        arr = x.numpy()
        assert isinstance(arr, np.ndarray)
        assert np.allclose(arr, np.array([1, 2, 3]))


class TestArithmetic:
    def test_addition(self):
        x = Tensor([1.0, 2.0, 3.0])
        y = Tensor([4.0, 5.0, 6.0])
        z = x + y
        assert np.allclose(z.data, np.array([5.0, 7.0, 9.0]))

    def test_subtraction(self):
        x = Tensor([5.0, 6.0, 7.0])
        y = Tensor([1.0, 2.0, 3.0])
        z = x - y
        assert np.allclose(z.data, np.array([4.0, 4.0, 4.0]))

    def test_multiplication(self):
        x = Tensor([1.0, 2.0, 3.0])
        y = Tensor([2.0, 3.0, 4.0])
        z = x * y
        assert np.allclose(z.data, np.array([2.0, 6.0, 12.0]))

    def test_division(self):
        x = Tensor([4.0, 6.0, 8.0])
        y = Tensor([2.0, 3.0, 4.0])
        z = x / y
        assert np.allclose(z.data, np.array([2.0, 2.0, 2.0]))

    def test_power(self):
        x = Tensor([2.0, 3.0, 4.0])
        y = Tensor([2.0, 2.0, 2.0])
        z = x ** y
        assert np.allclose(z.data, np.array([4.0, 9.0, 16.0]))

    def test_negation(self):
        x = Tensor([1.0, -2.0, 3.0])
        y = -x
        assert np.allclose(y.data, np.array([-1.0, 2.0, -3.0]))

    def test_scalar_operations(self):
        x = Tensor([1.0, 2.0, 3.0])
        
        # Scalar addition
        y = x + 5
        assert np.allclose(y.data, np.array([6.0, 7.0, 8.0]))
        
        # Scalar multiplication
        z = x * 2
        assert np.allclose(z.data, np.array([2.0, 4.0, 6.0]))
        
        # Right operations
        w = 5 + x
        assert np.allclose(w.data, np.array([6.0, 7.0, 8.0]))


class TestShapeOperations:
    def test_reshape(self):
        x = Tensor(np.arange(12))
        y = x.reshape((3, 4))
        assert y.shape == (3, 4)
        assert np.allclose(y.data, np.arange(12).reshape(3, 4))

    def test_transpose(self):
        x = Tensor(np.arange(6).reshape(2, 3))
        y = x.transpose((1, 0))
        assert y.shape == (3, 2)
        assert np.allclose(y.data, np.arange(6).reshape(2, 3).T)

    def test_broadcast(self):
        x = Tensor([[1.0], [2.0]])
        y = x.broadcast_to((2, 3))
        assert y.shape == (2, 3)
        expected = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        assert np.allclose(y.data, expected)

    def test_sum(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Sum all
        y = x.sum()
        assert y.data.shape == ()
        assert np.isclose(y.data, 10.0)
        
        # Sum along axis
        z = x.sum((0,))
        assert np.allclose(z.data, np.array([4.0, 6.0]))


class TestMatrixOperations:
    def test_matmul(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = Tensor([[5.0, 6.0], [7.0, 8.0]])
        z = x @ y
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(z.data, expected)

    def test_matmul_shapes(self):
        x = Tensor(np.random.randn(3, 4))
        y = Tensor(np.random.randn(4, 5))
        z = x @ y
        assert z.shape == (3, 5)


class TestActivations:
    def test_relu(self):
        from kuttygrad.ops import ReLU
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = ReLU()(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert np.allclose(y.data, expected)

    def test_sigmoid(self):
        from kuttygrad.ops import Sigmoid
        x = Tensor([0.0])
        y = Sigmoid()(x)
        assert np.isclose(y.data, 0.5)

    def test_tanh(self):
        from kuttygrad.ops import TanH
        x = Tensor([0.0])
        y = TanH()(x)
        assert np.isclose(y.data, 0.0)


class TestGradients:
    def test_simple_gradient(self):
        x = Tensor([2.0], requires_grad=True)
        y = x ** 2
        y.backward()
        assert np.isclose(x.grad.data, 4.0)

    def test_chain_rule(self):
        x = Tensor([2.0], requires_grad=True)
        y = x ** 2
        z = y * 3
        z.backward()
        # dz/dx = dz/dy * dy/dx = 3 * 2x = 6 * 2 = 12
        assert np.isclose(x.grad.data, 12.0)

    def test_multiple_parents(self):
        x = Tensor([2.0], requires_grad=True)
        y = x ** 2
        z = x * 3
        w = y + z
        w.backward()
        # dw/dx = dy/dx + dz/dx = 2x + 3 = 4 + 3 = 7
        assert np.isclose(x.grad.data, 7.0)

    def test_addition_gradient(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        z = x + y
        z.sum().backward()
        assert np.allclose(x.grad.data, np.ones(2))
        assert np.allclose(y.grad.data, np.ones(2))

    def test_multiplication_gradient(self):
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        z = x * y
        z.backward()
        assert np.isclose(x.grad.data, 3.0)
        assert np.isclose(y.grad.data, 2.0)


class TestGradcheck:
    def test_gradcheck_relu(self):
        def f(x):
            from kuttygrad.ops import ReLU
            return ReLU()(x).sum()
        
        x = Tensor(np.random.randn(2, 3), requires_grad=True)
        gradcheck_fn(f, x)

    def test_gradcheck_sigmoid(self):
        def f(x):
            from kuttygrad.ops import Sigmoid
            return Sigmoid()(x).sum()
        
        x = Tensor(np.random.randn(2, 3), requires_grad=True)
        gradcheck_fn(f, x)

    def test_gradcheck_matmul(self):
        # Use a fixed weight matrix
        w = Tensor(np.random.randn(3, 2), requires_grad=False)
        
        def f(x):
            return (x @ w).sum()
        
        x = Tensor(np.random.randn(2, 3), requires_grad=True)
        gradcheck_fn(f, x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
