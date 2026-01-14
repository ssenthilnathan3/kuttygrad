import numpy as np


def gradcheck(f, x, eps=1e-6, atol=1e-5, rtol=1e-4):
    """
    f : function that takes a Tensor and returns a scalar Tensor
    x : Tensor (requires_grad=True)
    """

    y = f(x)
    assert y.data.shape == (), "gradcheck requires scalar output"

    y.backward()
    grad_analytical = x.grad.data.copy()

    grad_numerical = np.zeros_like(x.data)

    it = np.nditer(x.data, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x.data[idx]

        # f(x + eps)
        x.data[idx] = orig + eps
        y_pos = f(x).data

        # f(x - eps)
        x.data[idx] = orig - eps
        y_neg = f(x).data

        x.data[idx] = orig

        grad_numerical[idx] = (y_pos - y_neg) / (2 * eps)
        it.iternext()

    if not np.allclose(grad_analytical, grad_numerical, atol=atol, rtol=rtol):
        print("❌ gradcheck failed")
        print("analytical:\n", grad_analytical)
        print("numerical:\n", grad_numerical)
        raise AssertionError("gradcheck failed")

    print("gradcheck passed")
