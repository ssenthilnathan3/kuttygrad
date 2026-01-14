from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class AddScalar(Function):
    def __init__(self, scalar):
        self.scalar = scalar

    @override
    def forward(self, *args: NDArray):
        (a,) = args
        return a + self.scalar

    @override
    def backward(self, *args):
        (out_grad,) = args
        return (out_grad,)
