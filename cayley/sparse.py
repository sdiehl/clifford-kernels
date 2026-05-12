import torch
from torch import Tensor


def dense_to_sparse_cayley(C: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Most entries of a Cayley tensor are zero, so we keep only the nonzero
    # coefficients and the (i, j, k) coordinates that locate them.
    if C.ndim != 3 or C.shape[0] != C.shape[1] or C.shape[0] != C.shape[2]:
        raise ValueError(f"expected cubic (n,n,n) tensor, got {tuple(C.shape)}")
    nz = torch.nonzero(C, as_tuple=False)
    ia = nz[:, 0].contiguous().to(torch.int32)
    ib = nz[:, 1].contiguous().to(torch.int32)
    ic = nz[:, 2].contiguous().to(torch.int32)
    sign = C[nz[:, 0], nz[:, 1], nz[:, 2]].contiguous().to(C.dtype)
    return ia, ib, ic, sign
