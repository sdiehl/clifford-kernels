# cayley-mlx

Sparse [Cayley table](https://en.wikipedia.org/wiki/Cayley_table) contraction in MLX.

```
uv sync
uv run pytest
```

```python
import mlx.core as mx
from cayley_mlx import sparse_cayley_from_sig
from cayley_mlx.metal import sparse_gp

ia, ib, ic, sign = sparse_cayley_from_sig(3, 0, 1)
x = mx.random.normal((32, 16))
y = mx.random.normal((32, 16))
out = sparse_gp(x, y, ia, ib, ic, sign)
```

`cayley_mlx.metal` is a Metal kernel with one thread per batch row accumulating into registers, differentiable through a custom VJP that reuses the kernel with permuted indices. `cayley_mlx.pure` is the same contraction as an MLX scatter-add that works under `mx.compile`.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE.md) file for details.
