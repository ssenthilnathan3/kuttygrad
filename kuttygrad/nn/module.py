from kuttygrad.tensor import Tensor


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self):
        """
        Returns an iterator over module parameters (Tensors with requires_grad=True).
        """
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and getattr(value, "requires_grad", False):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
            elif isinstance(value, (list, tuple)):
                for v in value:
                    if isinstance(v, Tensor) and getattr(v, "requires_grad", False):
                        yield v
                    elif isinstance(v, Module):
                        yield from v.parameters()

    def modules(self):
        """
        Returns an iterator over immediate child modules.
        """
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                yield value
            elif isinstance(value, (list, tuple)):
                for v in value:
                    if isinstance(v, Module):
                        yield v

    def train(self, mode: bool = True):
        """
        Sets the module in training mode.
        """
        self.training = mode
        for module in self.modules():
            module.train(mode)
        return self

    def eval(self):
        """
        Sets the module in evaluation mode.
        """
        return self.train(False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Override this method in subclasses.
        """
        raise NotImplementedError("Subclasses of Module must implement forward()")
