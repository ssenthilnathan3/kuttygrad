import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Exp(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        x = args[0]
        # cache the forward result so backward can reuse it
        self.y = numpy.exp(x)
        return self.y

    @override
    def backward(self, *args):
        (out_grad,) = args
        # derivative of exp(x) is exp(x); return as a single-element tuple
        return (out_grad * self.y,)


def exp(a: NDArray):
    return Exp()(a)
