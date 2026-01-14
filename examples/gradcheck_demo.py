"""
Gradient checking demo - verifying gradient correctness
"""
import numpy as np
from kuttygrad import Tensor, gradcheck_fn


def main():
    print("Gradient Checking Demo")
    print("=" * 50)
    
    # Example 1: Simple quadratic function
    print("\n1. Testing x^2:")
    def f1(x):
        return (x ** 2).sum()
    
    x1 = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    try:
        gradcheck_fn(f1, x1)
        print("   ✓ Gradcheck passed")
    except AssertionError:
        print("   ✗ Gradcheck failed")
    
    # Example 2: Matrix multiplication
    print("\n2. Testing matrix multiplication:")
    w2 = Tensor(np.random.randn(3, 2), requires_grad=False)
    def f2(x):
        return (x @ w2).sum()
    
    x2 = Tensor(np.random.randn(2, 3), requires_grad=True)
    try:
        gradcheck_fn(f2, x2)
        print("   ✓ Gradcheck passed")
    except AssertionError:
        print("   ✗ Gradcheck failed")
    
    # Example 3: ReLU activation
    print("\n3. Testing ReLU:")
    def f3(x):
        from kuttygrad.ops import ReLU
        return ReLU()(x).sum()
    
    x3 = Tensor(np.random.randn(2, 3), requires_grad=True)
    try:
        gradcheck_fn(f3, x3)
        print("   ✓ Gradcheck passed")
    except AssertionError:
        print("   ✗ Gradcheck failed")
    
    # Example 4: Sigmoid activation
    print("\n4. Testing Sigmoid:")
    def f4(x):
        from kuttygrad.ops import Sigmoid
        return Sigmoid()(x).sum()
    
    x4 = Tensor(np.random.randn(2, 3), requires_grad=True)
    try:
        gradcheck_fn(f4, x4)
        print("   ✓ Gradcheck passed")
    except AssertionError:
        print("   ✗ Gradcheck failed")
    
    # Example 5: Complex expression
    print("\n5. Testing complex expression (x^3 + 2*x):")
    def f5(x):
        return (x ** 3 + 2 * x).sum()
    
    x5 = Tensor(np.array([0.5, 1.5, 2.5]), requires_grad=True)
    try:
        gradcheck_fn(f5, x5)
        print("   ✓ Gradcheck passed")
    except AssertionError:
        print("   ✗ Gradcheck failed")
    
    print("\n" + "=" * 50)
    print("Gradient checking completed!")


if __name__ == "__main__":
    main()
