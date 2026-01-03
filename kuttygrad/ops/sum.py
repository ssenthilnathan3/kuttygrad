import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Summation(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        arr, axis = args
        self.axis = axis
        return numpy.sum(arr, axis)

    @override
    def backward(self, *args):
        out_grad = args[0]
        broadcasted_grad = out_grad.broadcast_to(self.axis)
        return broadcasted_grad, None
