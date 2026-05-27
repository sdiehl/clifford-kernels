import torch
from torch import Tensor


def _popcount(n: int) -> int:
    c = 0
    while n:
        n &= n - 1
        c += 1
    return c


def _swap_sign(a: int, b: int, dim: int) -> int:
    # Parity of swaps to reorder the concatenation of the two basis blades
    # (as bit-listed sequences) into canonical low-bit-first order.
    count = 0
    for j in range(dim):
        above = a >> (j + 1)
        if (b >> j) & 1:
            count += _popcount(above & ((1 << (dim - j - 1)) - 1))
    return 1 if count % 2 == 0 else -1


def _basis_sign(i: int, p: int, q: int) -> int:
    if i < p:
        return 1
    if i < p + q:
        return -1
    return 0  # null direction, squares to zero


def _metric_sign(mask: int, p: int, q: int, dim: int) -> int:
    # Product of basis-vector signs for every set bit of `mask`; zero if any
    # bit lands in the null block.
    acc = 1
    for i in range(dim):
        if (mask >> i) & 1:
            m = _basis_sign(i, p, q)
            if m == 0:
                return 0
            acc *= m
    return acc


def sparse_cayley_from_sig(
    p: int,
    q: int,
    r: int = 0,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Direct combinatorial construction of the sparse Cayley table for Cl(p,q,r).
    # Iterates over the (n^2) blade-mask pairs and emits (ia, ib, ic, sign) without
    # materialising the dense (n,n,n) tensor first. Cheap at small N, but at N=8
    # this avoids a 256 MB float32 allocation that dense_to_sparse_cayley would
    # otherwise need.
    dim = p + q + r
    n = 1 << dim
    ia_list: list[int] = []
    ib_list: list[int] = []
    ic_list: list[int] = []
    sgn_list: list[int] = []
    for a in range(n):
        for b in range(n):
            m = _metric_sign(a & b, p, q, dim)
            if m == 0:
                continue
            s = _swap_sign(a, b, dim) * m
            ia_list.append(a)
            ib_list.append(b)
            ic_list.append(a ^ b)
            sgn_list.append(s)
    ia = torch.tensor(ia_list, dtype=torch.int32)
    ib = torch.tensor(ib_list, dtype=torch.int32)
    ic = torch.tensor(ic_list, dtype=torch.int32)
    sign = torch.tensor(sgn_list, dtype=dtype)
    return ia, ib, ic, sign
