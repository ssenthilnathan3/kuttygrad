from typing import Any

import numpy

from kuttygrad.tensor import Tensor

from .module import Module
from . import functional as F


class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn  bias.
    Shape:
        - Input: `(batch_size, *, in_features)` where `*`
          means any number of additional dimensions.
        - Output: `(batch_size, *, out_features)` where
          all but the last dimension are the same shape as the input.

    Attributes:
        weight (Parameter): The learnable weights of the module of shape
                            `(in_features, out_features)`.
        bias (Parameter):   The learnable bias of the module
                             of shape `(out_features,)`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device: Any | None = None,
        dtype: str = "float32",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(
            numpy.random.randn(self.in_features, self.out_features) * 0.01,
            requires_grad=True,
        )
        if bias:
            self.bias = Tensor(
                numpy.zeros(self.out_features), requires_grad=True
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # Note: x.shape is (batch_size, in_features)
        # self.weight.shape is (in_features, out_features)
        # The result should have shape (batch_size, out_features)
        return F.linear(x, self.weight, self.bias)
