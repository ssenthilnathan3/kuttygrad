import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class ReLU(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        x = args[0]
        self.x = x
        return numpy.maximum(0, x)

    @override
    def backward(self, *args):
        (out_grad,) = args
        grad_x = out_grad * (self.x > 0)
        return (grad_x,)


def relu(a: NDArray):
    return ReLU()(a)
