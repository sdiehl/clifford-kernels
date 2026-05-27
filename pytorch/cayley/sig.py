from collections.abc import Iterator

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


def _cayley_entries(p: int, q: int, r: int) -> Iterator[tuple[int, int, int, int]]:
    # Yields (ia, ib, ic, sign) for every nonzero entry of the Cl(p,q,r)
    # geometric-product Cayley tensor. Iterates the n^2 blade-mask pairs and
    # skips entries that fall into the null block.
    dim = p + q + r
    n = 1 << dim
    for a in range(n):
        for b in range(n):
            m = _metric_sign(a & b, p, q, dim)
            if m == 0:
                continue
            yield a, b, a ^ b, _swap_sign(a, b, dim) * m


def sparse_cayley_from_sig(
    p: int,
    q: int,
    r: int = 0,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Direct combinatorial construction of the sparse Cayley table for Cl(p,q,r).
    # Emits (ia, ib, ic, sign) without materialising the dense (n,n,n) tensor
    # first; at N=8 this avoids a 256 MB float32 allocation.
    ia_l: list[int] = []
    ib_l: list[int] = []
    ic_l: list[int] = []
    sgn_l: list[int] = []
    for a, b, c, s in _cayley_entries(p, q, r):
        ia_l.append(a)
        ib_l.append(b)
        ic_l.append(c)
        sgn_l.append(s)
    return (
        torch.tensor(ia_l, dtype=torch.int32),
        torch.tensor(ib_l, dtype=torch.int32),
        torch.tensor(ic_l, dtype=torch.int32),
        torch.tensor(sgn_l, dtype=dtype),
    )


def dense_cayley_from_sig(
    p: int,
    q: int,
    r: int = 0,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    # Dense (n,n,n) Cayley tensor for Cl(p,q,r). Useful as a reference for the
    # einsum baseline; prefer sparse_cayley_from_sig when only the kernel
    # arrays are needed.
    n = 1 << (p + q + r)
    C = torch.zeros(n, n, n, dtype=dtype)
    for a, b, c, s in _cayley_entries(p, q, r):
        C[a, b, c] = s
    return C
