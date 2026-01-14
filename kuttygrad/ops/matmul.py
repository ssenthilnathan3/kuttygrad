import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class MatMul(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        arrA, arrB = args
        self.arrA = arrA
        self.arrB = arrB
        return numpy.matmul(arrA, arrB)

    @override
    def backward(self, *args):
        (out_grad,) = args
        # Gradient w.r.t arrA: out_grad @ arrB.T
        grad_a = numpy.matmul(out_grad, self.arrB.T)
        # Gradient w.r.t arrB: arrA.T @ out_grad
        grad_b = numpy.matmul(self.arrA.T, out_grad)
        return grad_a, grad_b
