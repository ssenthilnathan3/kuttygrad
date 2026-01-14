from __future__ import annotations

from numbers import Number
from typing import Any, Literal, Optional, TypeVar

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
        elif isinstance(data, (list, tuple)):
            self.data = np.array(data)
        elif isinstance(data, Number):
            self.data = np.array(data)
        else:
            raise TypeError(f"unsupported type for Tensor: {type(data)}")

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

        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Pow()(self, other)

    def __rpow__(self, other):
        """Right power: 5 ** tensor"""
        from .ops import Pow

        if not isinstance(other, Tensor):
            other = Tensor(other)
        # other is the base, self is the exponent
        return Pow()(other, self)

    def __sub__(self, other):
        """Subtraction: a - b"""
        from .ops import Sub

        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Sub()(self, other)

    def __rsub__(self, other):
        """Right subtraction: 5 - tensor"""
        from .ops import Sub

        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Sub()(other, self)

    def __truediv__(self, other):
        """Division: a / b"""

        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self * (other**-1)

    def __rtruediv__(self, other):
        """Right division: 5 / tensor"""
        return self.__truediv__(other)

    def __matmul__(self, other):
        """Matrix multiplication: a @ b"""
        from .ops import MatMul

        if not isinstance(other, Tensor):
            other = Tensor(other)
        return MatMul()(self, other)

    def __rmatmul__(self, other):
        """Right matrix multiplication: array @ tensor"""
        return self.__matmul__(other)

    def matmul(self, other):
        """Matrix multiplication method: a.matmul(b)"""
        return self.__matmul__(other)

    def sum(self, axes: Optional[tuple] | None = None):
        """Sum of tensor elements along specified axis"""
        from .ops import Summation

        return Summation(axes)(self)

    def broadcast_to(self, shape: tuple):
        """Broadcast tensor to specified shape"""
        from .ops import BroadcastTo

        return BroadcastTo(shape)(self)

    def reshape(self, shape: tuple):
        """Reshape tensor to specified shape"""
        from .ops import Reshape

        return Reshape(shape)(self)

    def __neg__(self):
        """Negation: -tensor"""
        from .ops import Negate

        return Negate()(self)

    def transpose(self, axes: Optional[tuple] | None = None):
        """Transpose tensor"""
        from .ops import Transpose

        return Transpose(axes)(self)

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

    def backward(self, grad=None):
        """
        Compute gradients for all tensors in the computation graph that
        leads to `self` and store them on each tensor's `.grad` attribute.

        Notes:
        - We pass raw numpy arrays (NDArray) to each operation's `backward`
          implementation so operations can use NumPy directly.
        - All gradients stored in the intermediate `grads` dict are `Tensor`
          instances with `requires_grad=False` to avoid building new graph
          nodes during backpropagation.
        - Each `backward` implementation should return a tuple/list with one
          entry per input (use `None` for inputs that don't require a gradient).
        """
        grads = {}

        # seed gradient (as a Tensor without gradient tracking)
        if grad is None:
            grads[self] = Tensor(np.ones_like(self.data), requires_grad=False)
        else:
            # normalize user-provided gradient into a Tensor (no grad tracking)
            if isinstance(grad, Tensor):
                if grad.requires_grad:
                    grads[self] = Tensor(grad.data, requires_grad=False)
                else:
                    grads[self] = grad
            else:
                grads[self] = Tensor(grad, requires_grad=False)

        topo_order = []
        visited = set()

        def build_topo(node):
            if node in visited:
                return
            visited.add(node)

            if node._inputs is not None:
                for parent in node._inputs:
                    build_topo(parent)

            topo_order.append(node)

        build_topo(self)

        # reverse topo for backprop
        for node in reversed(topo_order):
            out_grad = grads.get(node)
            if out_grad is None:
                continue

            # leaf nodes have no backward
            if node._op is None:
                continue

            # pass a numpy array (NDArray) to op.backward
            out_grad_arr = out_grad.data if isinstance(out_grad, Tensor) else out_grad

            parent_grads = node._op.backward(out_grad_arr)

            # normalize single-array returns to a tuple/list
            if not isinstance(parent_grads, (tuple, list)):
                parent_grads = (parent_grads,)

            # distribute grads to parents
            for parent, g in zip(node._inputs, parent_grads):
                if g is None:
                    continue

                # normalize gradients into Tensor objects (no grad tracking)
                if isinstance(g, Tensor):
                    if g.requires_grad:
                        g_tensor = Tensor(g.data, requires_grad=False)
                    else:
                        g_tensor = g
                else:
                    g_tensor = Tensor(g, requires_grad=False)

                if parent in grads:
                    grads[parent] = grads[parent] + g_tensor
                else:
                    grads[parent] = g_tensor

        # store .grad (leave as Tensor with requires_grad=False)
        for node, g in grads.items():
            node.grad = g
