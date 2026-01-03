import numpy
from typing_extensions import override

from kuttygrad.function import Function
from kuttygrad.tensor import NDArray


class MatMul(Function):
    @override
    def forward(self, *args: NDArray) -> NDArray:
        arrA, arrB = args
        return numpy.matmul(arrA, arrB)

    @override
    def backward(self, *args):
        out_grad, node = args
        return out_grad, node


def matmul(arrA: NDArray, arrB: NDArray):
    return MatMul()(arrA, arrB)
