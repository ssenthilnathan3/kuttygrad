import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class BroadcastTo(Function):
    @override
    def forward(self, *args) -> NDArray:
        arr, shape = args
        return numpy.broadcast_to(arr, shape)

    @override
    def backward(self, *args):
        out_grad = args[0]
        out_grad = numpy.sum(out_grad)
        return out_grad, None
