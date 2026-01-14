import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Abs(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        x = args[0]
        self.x = x
        return numpy.abs(x)

    @override
    def backward(self, *args):
        (out_grad,) = args
        # derivative of abs(x) is sign(x); define derivative at x=0 as 0
        return (out_grad * numpy.sign(self.x),)
