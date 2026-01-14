import numpy
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
        (out_grad,) = args

        # gradient w.r.t. base: exponent * base^(exponent - 1)
        base_grad = self.exponent * (self.base ** (self.exponent - 1))

        # gradient w.r.t. exponent: base^exponent * log(base)
        # Note: this will be NaN if base <= 0 for non-integer exponents.
        exponent_grad = (self.base**self.exponent) * numpy.log(self.base)

        return base_grad * out_grad, exponent_grad * out_grad
