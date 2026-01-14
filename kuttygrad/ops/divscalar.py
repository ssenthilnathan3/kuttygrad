from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class DivScalar(Function):
    def __init__(self, scalar):
        self.scalar = scalar

    @override
    def forward(self, *args: NDArray):
        (a,) = args
        self.a = a
        return a / self.scalar

    @override
    def backward(self, *args):
        (out_grad,) = args

        # gradient w.r.t a
        grad_a = out_grad / self.scalar

        # scalar is a fixed parameter (not a Tensor input), so we only
        # return gradient w.r.t. the tensor input
        return (grad_a,)
