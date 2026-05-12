from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _sparse_gp_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        ia_ptr,
        ib_ptr,
        ic_ptr,
        sign_ptr,
        n_nz,
        batch_size,
        n_blades,
        BLOCK_BATCH: tl.constexpr,
    ):
        # One program handles a contiguous slice of batch indices and walks the
        # entire list of nonzero Cayley entries, accumulating into out.
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        mask = offs < batch_size

        for k in range(0, n_nz):
            ia = tl.load(ia_ptr + k)
            ib = tl.load(ib_ptr + k)
            ic = tl.load(ic_ptr + k)
            s = tl.load(sign_ptr + k)

            x = tl.load(x_ptr + offs * n_blades + ia, mask=mask, other=0.0)
            y = tl.load(y_ptr + offs * n_blades + ib, mask=mask, other=0.0)
            prod = s * x * y

            # Atomic add since multiple k values may target the same ic, and
            # different programs may write to the same (batch, ic) slot only if
            # we partitioned by k; here we partition by batch, so within a
            # program a plain load/store would race across k. We accumulate
            # safely via atomic_add on the output element.
            tl.atomic_add(out_ptr + offs * n_blades + ic, prod, mask=mask)


def _reference_sparse_gp(
    x: Tensor,
    y: Tensor,
    ia: Tensor,
    ib: Tensor,
    ic: Tensor,
    sign: Tensor,
) -> Tensor:
    # scatter_add reference used when Triton is unavailable (e.g. macOS) and
    # also when inputs are on CPU; matches the kernel semantics exactly.
    n_blades = x.shape[-1]
    batch = x.shape[0]
    ia_l = ia.long()
    ib_l = ib.long()
    ic_l = ic.long()
    contrib = sign.to(x.dtype) * x[:, ia_l] * y[:, ib_l]
    out = torch.zeros(batch, n_blades, dtype=x.dtype, device=x.device)
    idx = ic_l.unsqueeze(0).expand(batch, -1)
    out.scatter_add_(1, idx, contrib)
    return out


def sparse_gp(
    x: Tensor,
    y: Tensor,
    ia: Tensor,
    ib: Tensor,
    ic: Tensor,
    sign: Tensor,
) -> Tensor:
    if x.shape != y.shape:
        raise ValueError(f"x and y must match, got {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.ndim != 2:
        raise ValueError(f"expected (batch, n_blades) inputs, got {tuple(x.shape)}")
    if not (ia.shape == ib.shape == ic.shape == sign.shape):
        raise ValueError("ia, ib, ic, sign must all have the same shape")

    # Fall back to torch on any path that cannot run the kernel: missing
    # triton, non-CUDA tensors, or empty sparsity (nothing to launch).
    if (not _HAS_TRITON) or (not x.is_cuda) or ia.numel() == 0:
        return _reference_sparse_gp(x, y, ia, ib, ic, sign)

    batch_size, n_blades = x.shape
    n_nz = ia.numel()

    x_c = x.contiguous()
    y_c = y.contiguous()
    out = torch.zeros_like(x_c)

    ia_c = ia.to(torch.int32).contiguous()
    ib_c = ib.to(torch.int32).contiguous()
    ic_c = ic.to(torch.int32).contiguous()
    sign_c = sign.to(x.dtype).contiguous()

    BLOCK_BATCH = 64
    grid = (triton.cdiv(batch_size, BLOCK_BATCH),)
    _sparse_gp_kernel[grid](
        x_c,
        y_c,
        out,
        ia_c,
        ib_c,
        ic_c,
        sign_c,
        n_nz,
        batch_size,
        n_blades,
        BLOCK_BATCH=BLOCK_BATCH,
    )
    return out
