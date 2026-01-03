import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class ReLU(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        x = args[0]
        return numpy.maximum(0, x)

    @override
    def backward(self, *args):
        out_grad, node = args
        return out_grad, node


def relu(a: NDArray):
    return ReLU()(a)
