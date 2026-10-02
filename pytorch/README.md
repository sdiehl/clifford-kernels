# cayley-triton

A tiny Triton kernel for the sparse [Cayley table](https://en.wikipedia.org/wiki/Cayley_table) contraction (the geometric product in Clifford algebras), differentiable, signature-agnostic.

The Cayley tensor $`C \in \mathbb{R}^{n \times n \times n}`$ of $`\mathrm{Cl}(p,q,r)`$ with $`n = 2^{p+q+r}`$ is roughly 90% zeros. This repo extracts the nonzero entries as `(ia, ib, ic, sign)` and runs a small Triton kernel that only touches them. Forward and backward share the kernel with permuted indices, so you can train through it.

```
uv sync
uv run pytest
```

```python
import torch
from cayley_torch import sparse_cayley_from_sig, sparse_gp

ia, ib, ic, sign = sparse_cayley_from_sig(8, 0, 0)  # Cl(8,0,0), 256 blades

x = torch.randn(32, 256, device="cuda", requires_grad=True)
y = torch.randn(32, 256, device="cuda", requires_grad=True)
out = sparse_gp(x, y, ia, ib, ic, sign)
out.sum().backward()
```

`sparse_cayley_from_sig` builds the index arrays directly from the signature; use `dense_to_sparse_cayley(C)` if you already have a dense Cayley tensor from elsewhere. Triton JITs on first launch and caches under `TRITON_CACHE_DIR`. Accumulation is in fp32 (fp64 for fp64 inputs). CPU tensors hit a `scatter_add` fallback that is numerically identical.

## `torch.compile`

`sparse_gp` is registered as the `cayley::sparse_gp` op via `torch.library.triton_op` with a registered autograd backward, so `torch.compile(..., fullgraph=True)` traces into the Triton kernel rather than treating it as opaque. See [`examples/torch_compile.py`](examples/torch_compile.py).

## HuggingFace `kernels`

Once published the kernel will be loadable from the Hub:

```python
from kernels import get_kernel

cayley = get_kernel("sdiehl/cayley-triton")
```

See [`examples/kernels_hub.py`](examples/kernels_hub.py) for the full call.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
