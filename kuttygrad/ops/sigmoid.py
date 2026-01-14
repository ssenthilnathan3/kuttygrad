import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Sigmoid(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        x = args[0]
        self.x = x
        self.y = 1 / (1 + numpy.exp(-x))
        return self.y

    @override
    def backward(self, *args):
        (out_grad,) = args
        return (out_grad * self.y * (1 - self.y),)


def sigmoid(a: NDArray):
    return Sigmoid()(a)
