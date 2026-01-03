from .tensor import NDArray, Tensor


class Function:
    def __call__(self, *inputs):
        requires_grad = any([i.requires_grad for i in inputs])
        inputs_data = [i.data for i in inputs]

        output_data = self.forward(*inputs_data)

        # wrap the output in tensor
        output_tensor = Tensor.from_data(data=output_data, requires_grad=requires_grad)

        if requires_grad:
            output_tensor._op = self
            output_tensor._inputs = inputs_data

        return output_tensor

    def forward(self, *args: NDArray) -> NDArray:
        """Computes the forward pass of the operation.
        Args:
            *args: One or more NumPy arrays
        """
        raise NotImplementedError()

    def backward(self, *args):
        """Calculates backward pass (gradients)
        Args:
            out_grad: upstream gradient flowing from output to input
            node: Value object holding inputs from forward pass
        """
        raise NotImplementedError()
