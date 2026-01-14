from typing import Optional

import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Transpose(Function):
    def __init__(self, axes: Optional[tuple] | None = None):
        self.axes = axes

    @override
    def forward(self, *args: NDArray) -> NDArray:
        (x,) = args
        self.x_shape = x.shape

        return numpy.transpose(x, self.axes)

    @override
    def backward(self, *args):
        (out_grad,) = args

        # no axes specified then reverse dimensions
        if self.axes is None:
            return (out_grad.T,)

        # inverse permutation
        inv_axes = numpy.argsort(self.axes)
        return (numpy.transpose(out_grad, inv_axes),)
