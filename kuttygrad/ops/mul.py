from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Mul(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        a, b = args
        self.a = a
        self.b = b
        return self.a * self.b

    @override
    def backward(self, *args):
        (out_grad,) = args

        grad_a = out_grad * self.b
        grad_b = out_grad * self.a

        return grad_a, grad_b
