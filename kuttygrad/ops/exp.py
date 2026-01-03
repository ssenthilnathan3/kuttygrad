import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Exp(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        x = args[0]
        return numpy.exp(x)

    @override
    def backward(self, *args):
        out_grad = args[0]
        return out_grad, None


def exp(a: NDArray):
    return Exp()(a)
