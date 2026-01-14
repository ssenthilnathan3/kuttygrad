import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class PowerScalar(Function):
    def __init__(self, scalar):
        self.scalar = scalar

    @override
    def forward(self, *args: NDArray):
        (a,) = args
        self.a = a
        # cache the forward result to reuse in backward if needed
        self.y = a**self.scalar
        return self.y

    @override
    def backward(self, *args):
        (out_grad,) = args

        grad_a = out_grad * self.scalar * (self.a ** (self.scalar - 1))

        # scalar is a fixed parameter (not a Tensor input), so only return
        # gradient w.r.t. the tensor input
        return (grad_a,)
