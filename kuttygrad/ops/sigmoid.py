import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Sigmoid(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        x = args[0]
        return 1 / (1 + numpy.exp(-x))

    @override
    def backward(self, *args):
        out_grad, node = args
        return out_grad, node


def sigmoid(a: NDArray):
    return Sigmoid()(a)
