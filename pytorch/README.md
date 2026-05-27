# cayley-triton

A tiny Triton kernel for the sparse [Cayley table](https://en.wikipedia.org/wiki/Cayley_table) contraction (the geometric product in Clifford algebras), differentiable, signature-agnostic. Part of my ongoing work on transformers over Clifford algebras.

The Cayley tensor $`C \in \mathbb{R}^{n \times n \times n}`$ of $`\mathrm{Cl}(p,q,r)`$ with $`n = 2^{p+q+r}`$ is roughly 90% zeros, so the geometric product $`(x * y)_k = \sum_{i,j} x_i y_j C_{ijk}`$ wastes most of its work in a dense Einstein summation. This repo extracts the nonzero entries as `(ia, ib, ic, sign)` and runs a small Triton kernel that only touches them. Forward and backward both compile to the same kernel walking permuted index arrays, so you can train through it.

```
uv sync
uv run pytest
```

```python
import torch
from cayley import sparse_cayley_from_sig, sparse_gp

# Build the sparse Cayley table directly from the signature. Avoids ever
# materialising the dense (n,n,n) tensor, which matters past N = 6 or so.
ia, ib, ic, sign = sparse_cayley_from_sig(8, 0, 0)  # Cl(8,0,0), 256 blades

x = torch.randn(32, 256, device="cuda", requires_grad=True)
y = torch.randn(32, 256, device="cuda", requires_grad=True)
out = sparse_gp(x, y, ia, ib, ic, sign)  # shape (32, 256)
out.sum().backward()  # backward kernel too
```

For algebras where you already have a dense Cayley tensor (e.g. coming from another GA library), `dense_to_sparse_cayley(C)` is the alternative path; `sparse_cayley_from_sig(p, q, r)` is preferable whenever you only need the kernel-ready arrays.

The default dense einsum spends ~90% of its FLOPs multiplying by structural zeros. This kernel skips them entirely and writes results with a single fused atomic-add per nonzero, so memory traffic and arithmetic both scale with the sparsity rather than $`n^3`$. That matters in training loops where the geometric product runs on every layer of every forward and backward pass and quickly dominates step time.

Triton compiles the kernel on first launch and caches the result. Set `TRITON_CACHE_DIR` to control where the cache lives if the default is inconvenient. CPU and non-CUDA tensors fall back automatically to a `scatter_add_` reference path that is numerically identical.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
