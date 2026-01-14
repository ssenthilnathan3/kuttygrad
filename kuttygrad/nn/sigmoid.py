from .module import Module


class Sigmoid(Module):
    """
    nn.Sigmoid module: applies the sigmoid function elementwise.
    Usage:
        sigmoid = nn.Sigmoid()
        out = sigmoid(x)
    """

    def forward(self, x):
        from kuttygrad.ops import Sigmoid as SigmoidOp

        return SigmoidOp()(x)
