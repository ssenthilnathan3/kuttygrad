from .module import Module


class Tanh(Module):
    """
    nn.Tanh module: applies the hyperbolic tangent function elementwise.
    Usage:
        tanh = nn.Tanh()
        out = tanh(x)
    """

    def forward(self, x):
        from kuttygrad.ops import TanH as TanhOp

        return TanhOp()(x)
