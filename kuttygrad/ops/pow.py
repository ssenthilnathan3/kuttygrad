from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class Pow(Function):
    @override
    def forward(self, *args) -> NDArray:
        base, exponent = args
        self.base = base
        self.exponent = exponent
        return base**exponent

    @override
    def backward(self, *args):
        out_grad = args[0]
        base_grad = self.exponent * (self.base ** (self.exponent - 1))
        return base_grad * out_grad
