# cayley-mlx

Sparse [Cayley table](https://en.wikipedia.org/wiki/Cayley_table) contraction in MLX.

```
uv sync
uv run pytest
```

```python
import mlx.core as mx
from cayley import sparse_cayley_from_sig

ia, ib, ic, sign = sparse_cayley_from_sig(3, 0, 1)
x = mx.random.normal((32, 16))
y = mx.random.normal((32, 16))
```

The Metal kernel uses `atomic_fetch_add_explicit` on float, which requires Metal 3.1+ (Apple GPU family M3 and later). On M1/M2 it will fail to compile.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE.md) file for details.
