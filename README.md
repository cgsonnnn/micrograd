# micrograd (practice)

A from-scratch practice implementation of a tiny scalar autograd engine, written while following Andrej Karpathy's video [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0).

The goal is to understand how backpropagation works: how a computation graph is built during the forward pass, and how gradients flow back through it via the chain rule.

## What's implemented

**`Value` class** — wraps a single scalar and records how it was computed.

| Feature | Details |
|---|---|
| Attributes | `data`, `grad`, `_prev` (parent nodes), `_op` (operation) |
| Operations | `+`, `*`, `**` (constant exponent), unary `-`, `-`, `/`, `exp()` |
| Mixed with numbers | `a + 1`, `2 * a`, etc. via `__radd__` / `__rmul__` |
| Backpropagation | `backward()` uses a topological sort, then calls each node's `_backward` in reverse order |

**Helpers and exercises**

- `numgrad(f, inputs)` — numerical gradient check with finite differences, used to verify `backward()`
- `grad_des(...)` — gradient descent on `(w - 3)**2`
- `grad_des2(...)` — gradient descent on `(a*b - 6)**2 + (a + b - 5)**2`, which has two solutions, (2, 3) and (3, 2)

## Usage

```python
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
L = (a * b + c) ** 2
L.backward()
print(L.data, a.grad, b.grad, c.grad)   # 16.0 -24.0 16.0 8.0
```

Run interactively:

```bash
python -i micrograd.py
>>> grad_des(30, 5, 0.1)
>>> grad_des2(20000, 1.0, 1.5, 0.01)
```

## Notes from the experiments

- **Learning rate:** for `(w - 3)**2`, each step multiplies the distance to the minimum by `1 - 2*lr`. It converges for `0 < lr < 1`, oscillates forever at `lr = 1`, and diverges above that.
- **Saddle point:** in `grad_des2`, starting from `a = b` keeps the two parameters identical forever, so the loss gets stuck at about 0.0088 on a saddle point instead of reaching 0. A tiny asymmetry (`b = 1.001`) eventually escapes, but takes roughly 12,000 steps at `lr = 0.01`. This is why neural network weights are initialized randomly.
- **Zeroing gradients:** gradients accumulate with `+=`. That is required within one backward pass, but they must be reset to 0 between training steps.

## Credits

Based on [karpathy/micrograd](https://github.com/karpathy/micrograd) (MIT License, Copyright (c) 2020 Andrej Karpathy).
