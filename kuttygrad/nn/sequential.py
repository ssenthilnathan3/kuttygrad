from kuttygrad import Tensor

from .module import Module


class Sequential(Module):
    """A container that chains a sequence of modules together.
    Example:
        # A simple 2-layer MLP for MNIST
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        logits = model(input_tensor)
    """

    def __init__(self, *modules: list[Module]):
        super().__init__()
        self._modules = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        for mod in self._modules:
            x = mod(x)
        return x
