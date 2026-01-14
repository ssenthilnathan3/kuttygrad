from math import prod

from .module import Module


class Flatten(Module):
    """
    nn.Flatten module: Flattens a tensor by reshaping it to `(batch_size, -1)`.
    Usage:
        flatten = nn.Flatten()
        out = flatten(x)
    """

    def forward(self, x):
        from kuttygrad.ops import Reshape as ReshapeOp

        shape = x.shape
        batch_size = shape[0]
        return ReshapeOp((batch_size, -1))(x)
