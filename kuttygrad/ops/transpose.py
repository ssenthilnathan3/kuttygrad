import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Transpose(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        x = args[0]
        self.x = x
        return numpy.transpose(x)

    @override
    def backward(self, *args):
        out_grad = args[0]
        return out_grad.T
