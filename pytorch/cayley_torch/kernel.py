from __future__ import annotations

import torch
from torch import Tensor
from torch.library import triton_op, wrap_triton

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
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
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        mask = offs < batch_size
        row = offs * n_blades
        acc_ty = out_ptr.dtype.element_ty

        for k in range(0, n_nz):
            ia = tl.load(ia_ptr + k)
            ib = tl.load(ib_ptr + k)
            ic = tl.load(ic_ptr + k)
            s = tl.load(sign_ptr + k)

            x = tl.load(x_ptr + row + ia, mask=mask, other=0.0).to(acc_ty)
            y = tl.load(y_ptr + row + ib, mask=mask, other=0.0).to(acc_ty)

            # Each lane owns its output row and k runs in order, so no atomics.
            o = tl.load(out_ptr + row + ic, mask=mask, other=0.0)
            tl.store(out_ptr + row + ic, o + s * x * y, mask=mask)


def _reference_sparse_gp(
    x: Tensor,
    y: Tensor,
    ia: Tensor,
    ib: Tensor,
    ic: Tensor,
    sign: Tensor,
) -> Tensor:
    ia_l, ib_l, ic_l = ia.long(), ib.long(), ic.long()
    contrib = sign.to(x.dtype) * x[:, ia_l] * y[:, ib_l]
    idx = ic_l.unsqueeze(0).expand(x.shape[0], -1)
    return torch.zeros_like(x).scatter_add(1, idx, contrib)


@triton_op("cayley::sparse_gp", mutates_args=())
def _sparse_gp_op(
    x: Tensor,
    y: Tensor,
    ia: Tensor,
    ib: Tensor,
    ic: Tensor,
    sign: Tensor,
) -> Tensor:
    if not (_HAS_TRITON and x.is_cuda) or ia.numel() == 0:
        return _reference_sparse_gp(x, y, ia, ib, ic, sign)

    batch_size, n_blades = x.shape
    acc_dtype = torch.promote_types(x.dtype, torch.float32)
    out = torch.zeros(batch_size, n_blades, dtype=acc_dtype, device=x.device)

    BLOCK_BATCH = 128
    grid = (triton.cdiv(batch_size, BLOCK_BATCH),)
    wrap_triton(_sparse_gp_kernel)[grid](
        x.contiguous(),
        y.contiguous(),
        out,
        ia.to(torch.int32).contiguous(),
        ib.to(torch.int32).contiguous(),
        ic.to(torch.int32).contiguous(),
        sign.to(acc_dtype).contiguous(),
        ia.numel(),
        batch_size,
        n_blades,
        BLOCK_BATCH=BLOCK_BATCH,
        num_warps=4,
    )
    return out.to(x.dtype)


def _sparse_gp_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _sparse_gp_backward(ctx, dout):
    x, y, ia, ib, ic, sign = ctx.saved_tensors
    dx = dy = None
    if ctx.needs_input_grad[0]:
        dx = torch.ops.cayley.sparse_gp(y, dout, ib, ic, ia, sign)
    if ctx.needs_input_grad[1]:
        dy = torch.ops.cayley.sparse_gp(x, dout, ia, ic, ib, sign)
    return dx, dy, None, None, None, None


_sparse_gp_op.register_autograd(_sparse_gp_backward, setup_context=_sparse_gp_setup_context)


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
    if not (x.device == y.device == ia.device == ib.device == ic.device == sign.device):
        raise ValueError("all inputs must be on the same device")

    return torch.ops.cayley.sparse_gp(x, y, ia, ib, ic, sign)
