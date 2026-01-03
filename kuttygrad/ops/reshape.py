from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Reshape(Function):
    """
    arr: input array to reshape
    shape: output dim
    """

    @override
    def forward(self, *args) -> NDArray:
        arr, shape = args
        self.shape = shape
        return arr.reshape(shape)

    @override
    def backward(self, *args):
        out_grad = args[0]
        out_grad_reshaped = out_grad.reshape(self.shape)
        return out_grad_reshaped, None
