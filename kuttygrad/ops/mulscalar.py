from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class MulScalar(Function):
    def __init__(self, scalar):
        self.scalar = scalar

    @override
    def forward(self, *args: NDArray):
        (a,) = args
        self.a = a
        return a * self.scalar

    @override
    def backward(self, *args):
        (out_grad,) = args
        grad_a = out_grad * self.scalar
        return (grad_a,)
