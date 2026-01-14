from .module import Module


class ReLU(Module):
    """
    nn.ReLU module: applies the rectified linear unit function elementwise.
    Usage:
        relu = nn.ReLU()
        out = relu(x)
    """

    def forward(self, x):
        from kuttygrad.ops import ReLU as ReLUOp

        return ReLUOp()(x)
