# cayley-triton

A tiny triton kernel for the sparse [Cayley table](https://en.wikipedia.org/wiki/Cayley_table) contraction (the geometric product in clifford algebras). Part of my ongoing work on transformers over geometric algebras.

The Cayley tensor $`C \in \mathbb{R}^{n \times n \times n}`$ of $`\mathrm{Cl}(p,q,r)`$ with $`n = 2^{p+q+r}`$ is roughly 90% zeros, so the geometric product $`(x * y)_k = \sum_{i,j} x_i y_j C_{ijk}`$ wastes most of its work in a dense Einstein summation. This repo extracts the nonzero entries as `(ia, ib, ic, sign)` and runs a small Triton kernel that only touches them.

```
uv sync
uv run pytest
```

The default dense einsum spends ~90% of its FLOPs multiplying by structural zeros. So we built this custom kernel that skips them entirely and writes results with a single fused atomic-add per nonzero, so memory traffic and arithmetic both scale with the sparsity rather than $n^3$, which matters in training loops where the geometric product runs on every layer of every forward and backward pass and quickly dominates step time.

```python
import torch
from cayley import dense_to_sparse_cayley, sparse_cayley_from_sig, sparse_gp

# Build a dense Cayley tensor however you like. For Cl(3,0,1) you can
# lift the construction from any Clifford-algebra library, or write your own.
C = build_cayley(p=3, q=0, r=1)  # shape (16, 16, 16)
ia, ib, ic, sign = dense_to_sparse_cayley(C)

x = torch.randn(32, 16, device="cuda")
y = torch.randn(32, 16, device="cuda")
out = sparse_gp(x, y, ia, ib, ic, sign)  # shape (32, 16)
```

For larger algebras (e.g. $`\mathrm{Cl}(8,0,0)`$ with $`n = 256`$), materialising the full $`n \times n \times n`$ Cayley tensor costs hundreds of MB. The `sparse_cayley_from_sig` constructor builds the table combinatorially from $`(p, q, r)`$ directly, never allocating the dense form:

```python
from cayley import sparse_cayley_from_sig, sparse_gp

ia, ib, ic, sign = sparse_cayley_from_sig(8, 0, 0)  # Cl(8,0,0), 256 blades
x = torch.randn(32, 256, device="cuda")
y = torch.randn(32, 256, device="cuda")
out = sparse_gp(x, y, ia, ib, ic, sign)
```

`dense_to_sparse_cayley` stays as the right tool when you already have a dense tensor lying around (e.g. coming from another GA library); `sparse_cayley_from_sig` is preferable whenever you only need the kernel-ready arrays.

Triton compiles the kernel on first launch and caches the result. Set `TRITON_CACHE_DIR` to control where the cache lives if the default is inconvenient.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
