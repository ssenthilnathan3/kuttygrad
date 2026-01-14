import numpy
from numpy.typing import NDArray
from typing_extensions import override

from kuttygrad.function import Function


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    @override
    def forward(self, *args) -> NDArray:
        (arr,) = args
        self.input_shape = arr.shape
        return numpy.broadcast_to(arr, self.shape)

    @override
    def backward(self, *args):
        (out_grad,) = args

        grad = out_grad

        # 1. if input had fewer dims, sum over the leading axes
        ndim_diff = grad.ndim - len(self.input_shape)
        if ndim_diff > 0:
            grad = numpy.sum(grad, axis=tuple(range(ndim_diff)))

        # 2. sum over axes where input shape was 1
        for i, (in_dim, out_dim) in enumerate(zip(self.input_shape, grad.shape)):
            if in_dim == 1 and out_dim > 1:
                grad = numpy.sum(grad, axis=i, keepdims=True)

        # 3. final shape must match input exactly
        grad = grad.reshape(self.input_shape)

        return (grad,)
