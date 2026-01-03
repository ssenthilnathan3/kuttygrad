import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Negate(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        x = args[0]
        return numpy.negative(x)

    @override
    def backward(self, *args):
        out_grad = args[0]
        return -out_grad
