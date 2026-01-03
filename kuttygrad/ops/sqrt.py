import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Sqrt(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        x = args[0]
        self.x = x
        return numpy.sqrt(x)

    @override
    def backward(self, *args):
        out_grad = args[0]
        return out_grad * (1 / (2 * self.x))
