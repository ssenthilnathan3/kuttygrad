import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Add(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        a, b = args
        self.a_shape = a.shape
        self.b_shape = b.shape
        return a + b

    @override
    def backward(self, *args):
        (out_grad,) = args
        
        # Handle broadcasting for gradient of a
        grad_a = out_grad
        ndim_diff = grad_a.ndim - len(self.a_shape)
        if ndim_diff > 0:
            grad_a = numpy.sum(grad_a, axis=tuple(range(ndim_diff)))
        for i, (in_dim, out_dim) in enumerate(zip(self.a_shape, grad_a.shape)):
            if in_dim == 1 and out_dim > 1:
                grad_a = numpy.sum(grad_a, axis=i, keepdims=True)
        grad_a = grad_a.reshape(self.a_shape)
        
        # Handle broadcasting for gradient of b
        grad_b = out_grad
        ndim_diff = grad_b.ndim - len(self.b_shape)
        if ndim_diff > 0:
            grad_b = numpy.sum(grad_b, axis=tuple(range(ndim_diff)))
        for i, (in_dim, out_dim) in enumerate(zip(self.b_shape, grad_b.shape)):
            if in_dim == 1 and out_dim > 1:
                grad_b = numpy.sum(grad_b, axis=i, keepdims=True)
        grad_b = grad_b.reshape(self.b_shape)
        
        return grad_a, grad_b
