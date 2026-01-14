from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Reshape(Function):
    """
    arr: input array to reshape
    shape: output dim
    """

    def __init__(self, shape):
        self.shape = shape

    @override
    def forward(self, *args) -> NDArray:
        (arr,) = args
        self.old_shape = arr.shape
        return arr.reshape(self.shape)

    @override
    def backward(self, *args):
        (out_grad,) = args
        out_grad_reshaped = out_grad.reshape(self.old_shape)
        return (out_grad_reshaped,)
