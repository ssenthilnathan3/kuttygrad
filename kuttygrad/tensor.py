from typing import Any, Literal, TypeVar

import numpy as np
from typing_extensions import assert_never

NDArray = np.ndarray
ScalarType = np.generic


T = TypeVar("T", bound="Tensor")


class Tensor:
    _op: Any | None
    _inputs: Any | None

    def __init__(
        self,
        data: list | ScalarType | NDArray | "Tensor",
        *,
        device: Literal["cpu", "gpu"] | None = None,
        dtype="float32",
        requires_grad=True,
    ):
        if isinstance(data, Tensor):
            self.data: NDArray = data.data
        elif isinstance(data, NDArray):
            self.data = data
        elif isinstance(data, list) or isinstance(data, ScalarType):
            self.data = np.array(data)
        else:
            assert_never(data)

        self._device: Literal["cpu", "gpu"] = device or "cpu"
        self.requires_grad = requires_grad
        self._op = None
        self._inputs = None

    @classmethod
    def from_data(cls, data, *, device=None, dtype="float32", requires_grad=True):
        return cls(
            data,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __str__(self):
        """
        Simple string representation (just the data).
        Example:
            >>> x = Tensor([1, 2, 3])
            >>> print(x)
            [1. 2. 3.]
        """
        return str(self.data)

    def __add__(self, other):
        """Addition: a + b"""
        from .ops import Add

        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Add()(self, other)

    def __radd__(self, other):
        """Right addition: 5 + tensor"""
        return self.__add__(other)

    def __mul__(self, other):
        """Multiplication: a * b"""
        from .ops import Mul

        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Mul()(self, other)

    def __rmul__(self, other):
        """Right multiplication: 5 * tensor"""
        return self.__mul__(other)

    def __pow__(self, other):
        """Power: base ** exponent"""
        from .ops import Pow

        return Pow()(self, other)

    def __rpow__(self, other):
        """Right power: 5 ** tensor"""
        return self.__pow__(other)

    # def __pow__(self, other):
    #     """Power: base ** exponent"""
    #     from .ops import Pow

    #     return Pow()(self, other)

    # def __rpow__(self, other):
    #     """Right power: 5 ** tensor"""
    #     return self.__pow__(other)

    @property
    def shape(self):
        """Shape of the Tensor"""
        return self.data.shape

    @property
    def dtype(self):
        """Data type of the Tensor"""
        return self.data.dtype

    @property
    def device(self) -> Literal["cpu", "gpu"] | None:
        """Device where the tensor lives"""
        return self._device

    @property
    def ndim(self):
        """No of dimensions of the Tensor"""
        return self.data.ndim

    @property
    def size(self):
        """Size of the Tensor"""
        return self.data.size

    def numpy(self):
        """
        Return the data as a NumPy array (detached from the autograd graph).
        This returns a copy, so modifying the result will not affect
        the tensor's data.
        Examples:
            >>> x = Tensor([1, 2, 3])
            >>> y = x + 1   # y is still a Tensor, part of the graph
            >>> z = x.numpy() + 1  # z is a NumPy array, not part of the graph
        Returns:
            np.ndarray: A copy of the tensor's data as a NumPy array.
        """
        return self.data.copy()

    def detach(self):
        """
        Creates a new Tensor with same data but no gradient tracking.
        Useful when you want to use values without building
        computation graph.
        Returns:
            Tensor: New tensor with requires_grad=False
        Example:
            >>> x = Tensor([1, 2, 3], requires_grad=True)
            >>> y = x.detach()  # y doesn't track gradients
            >>> z = y * 2       # This operation won't be in graph
        """
        return Tensor(self.data, requires_grad=False)

    def backward(self, out_grad=None):
        pass
