import torch
from torch import Tensor


def dense_to_sparse_cayley(C: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if C.ndim != 3 or C.shape[0] != C.shape[1] or C.shape[0] != C.shape[2]:
        raise ValueError(f"expected cubic (n,n,n) tensor, got {tuple(C.shape)}")
    ia, ib, ic = torch.nonzero(C, as_tuple=True)
    sign = C[ia, ib, ic].contiguous().to(C.dtype)
    return (
        ia.contiguous().to(torch.int32),
        ib.contiguous().to(torch.int32),
        ic.contiguous().to(torch.int32),
        sign,
    )
