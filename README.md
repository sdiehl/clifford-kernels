# cayley-triton

A tiny triton kernel for the sparse cayley table contraction (the geometric product in clifford algebras).

The Cayley tensor $`C \in \mathbb{R}^{n \times n \times n}`$ of $`\mathrm{Cl}(p,q,r)`$ with $`n = 2^{p+q+r}`$ is roughly 90% zeros, so the geometric product $`(x * y)_k = \sum_{i,j} x_i y_j C_{ijk}`$ wastes most of its work in a dense einsum. This repo extracts the nonzero entries as `(ia, ib, ic, sign)` and runs a small Triton kernel that only touches them.

```
uv sync
uv run pytest
```

```python
import torch
from tiny_cayley import dense_to_sparse_cayley, sparse_gp

# Build a dense Cayley tensor however you like; for Cl(3,0,1) you can lift
# the construction from tiny-gatr's Sig.structure(), or write your own.
C = build_cayley(p=3, q=0, r=1)  # shape (16, 16, 16)
ia, ib, ic, sign = dense_to_sparse_cayley(C)

x = torch.randn(32, 16, device="cuda")
y = torch.randn(32, 16, device="cuda")
out = sparse_gp(x, y, ia, ib, ic, sign)  # shape (32, 16)
```

Triton compiles the kernel on first launch and caches the result. Set `TRITON_CACHE_DIR` to control where the cache lives if the default is inconvenient.
