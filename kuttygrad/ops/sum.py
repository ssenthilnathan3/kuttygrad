from typing import Optional

import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Summation(Function):
    def __init__(self, axes: Optional[tuple] | None = None):
        self.axes = axes

    @override
    def forward(self, *args: NDArray) -> NDArray:
        (arr,) = args
        self.arr_shape = arr.shape
        return numpy.sum(arr, axis=self.axes)

    @override
    def backward(self, *args):
        (out_grad,) = args

        grad = out_grad

        # normalize axes
        if self.axes is None:
            axes = tuple(range(len(self.arr_shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes

        # insert summed axes back as size-1 dims
        for ax in sorted(axes):
            grad = numpy.expand_dims(grad, axis=ax)

        # broadcast to input shape
        grad = numpy.broadcast_to(grad, self.arr_shape)

        return (grad,)
