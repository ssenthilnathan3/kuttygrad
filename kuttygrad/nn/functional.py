"""
Functional API for kuttygrad, similar to torch.nn.functional (F).
Provides stateless layer operations as functions.
"""

from kuttygrad.tensor import Tensor


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """
    Applies a linear transformation to the incoming data: y = x @ W + b

    Args:
        input (Tensor): Input tensor of shape (batch, in_features)
        weight (Tensor): Weight tensor of shape (in_features, out_features)
        bias (Tensor, optional): Bias tensor of shape (out_features,). Default: None

    Returns:
        Tensor: Output tensor of shape (batch, out_features)
    """
    # weight: (in_features, out_features)
    # input: (batch, in_features)
    # output: (batch, out_features)
    output = input @ weight
    if bias is not None:
        output = output + bias
    return output
