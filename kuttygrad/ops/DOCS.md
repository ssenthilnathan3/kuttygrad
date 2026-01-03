# finding the backward function of any operation or function

### base formula

```python
  backward = out_grad * loc_derivative
```
## local derivative

> *if i change the input by a tiny amount (`dx`), how much does the output (`dy`) change?*

## 1. local derivative intuition

### simple transformations (math)

* **(x^2)** → change is **(2x)**
* **(1/x)** → change is **(-1/x^2)**
* **(\ln(x))** → change is **(1/x)**
* **(\sqrt{x})** → change is **(1/(2\sqrt{x}))**
* **(e^x)** → change is **(e^x)**

### linear / structural ops (data movement)

> no math change, only *where* values live

* **transpose**
* **reshape**
* **roll / permute**

local derivative is basically **1**, but mapped to a different position.

### logic-based ops (gates)

> data either passes or gets blocked

* **relu** -> `1` where `x > 0`, else `0`
* **max** -> `1` for the winner, `0` for others

## 2. apply the incoming gradient (chain rule)

in backward, you always receive:

[
\text{out_grad} = \frac{dL}{dy}
]

to compute input gradient:

[
\frac{dL}{dx} = \text{out_grad} \times \text{local derivative}
]

### mental formula

```
backward = out_grad * local_derivative
```

no exceptions.

## 3. shape integrity check

> **the gradient of an input must have the exact same shape as the input**

use this to verify correctness.

### common cases

* **transpose forward**
  → transpose gradient back

* **reshape forward**
  → reshape gradient back

* **sum forward (many → one)**
  → broadcast gradient back to input shape

* **broadcast forward (one → many)**
  → sum gradient back down

if shapes don’t match, your backward is wrong.

## practice examples

| operation  | forward    | local derivative | mental backward                   |
| ---------- | ---------- | ---------------- | --------------------------------- |
| reciprocal | (1/x)      | (-1/x^2)         | `out_grad * (-1 / x**2)`          |
| negative   | (-x)       | (-1)             | `out_grad * -1`                   |
| sqrt       | (\sqrt{x}) | (1 / (2√x))      | `out_grad / (2 * sqrt(x))`        |
| exp        | (e^x)      | (e^x)            | `out_grad * exp(x)`               |
| sum        | (\sum x)   | `1` (for all)    | broadcast `out_grad` to `x.shape` |

## the gold rule

> **if you’re stuck:**
>
> 1. look at the *forward* operation
> 2. undo the *movement* (shape-wise)
> 3. multiply by the calculus derivative

**move out_grad from `y`-space back to `x`-space.**

that’s autodiff.
