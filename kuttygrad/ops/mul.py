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
        out_grad, out_grad = args
        return self.a * out_grad, self.b * out_grad
