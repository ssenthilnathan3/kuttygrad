import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Add(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        a, b = args
        return a + b

    @override
    def backward(self, *args):
        out_grad, out_grad = args
        return out_grad, out_grad


def add(a: NDArray | numpy.generic, b: NDArray | numpy.generic):
    return Add()(a, b)
