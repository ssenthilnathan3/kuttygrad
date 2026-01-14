import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class TanH(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        x = args[0]
        self.y = numpy.tanh(x)
        return self.y

    @override
    def backward(self, *args):
        (out_grad,) = args
        return (out_grad * (1 - self.y**2),)


def tanh(a: NDArray):
    return TanH()(a)
