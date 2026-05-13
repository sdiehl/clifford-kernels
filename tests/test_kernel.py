import torch

from cayley import dense_to_sparse_cayley, sparse_cayley_from_sig, sparse_gp


def _popcount(n: int) -> int:
    c = 0
    while n:
        n &= n - 1
        c += 1
    return c


def _swap_sign(a: int, b: int, dim: int) -> int:
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
    return 0


def _metric_sign(mask: int, p: int, q: int, dim: int) -> int:
    acc = 1
    for i in range(dim):
        if (mask >> i) & 1:
            m = _basis_sign(i, p, q)
            if m == 0:
                return 0
            acc *= m
    return acc


def cayley_dense(p: int, q: int, r: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    dim = p + q + r
    n = 1 << dim
    sign = torch.zeros(n, n, dtype=torch.int64)
    idx = torch.zeros(n, n, dtype=torch.int64)
    for a in range(n):
        for b in range(n):
            m = _metric_sign(a & b, p, q, dim)
            if m == 0:
                continue
            sign[a, b] = _swap_sign(a, b, dim) * m
            idx[a, b] = a ^ b
    C = torch.zeros(n, n, n, dtype=dtype)
    aa = torch.arange(n).view(n, 1).expand(n, n)
    bb = torch.arange(n).view(1, n).expand(n, n)
    C[aa, bb, idx] = sign.to(dtype)
    return C


def _check_signature(p: int, q: int, r: int = 0, batch: int = 4, seed: int = 0) -> None:
    torch.manual_seed(seed)
    C = cayley_dense(p, q, r)
    n = C.shape[0]
    ia, ib, ic, sign = dense_to_sparse_cayley(C)

    x = torch.randn(batch, n)
    y = torch.randn(batch, n)

    expected = torch.einsum("bi,bj,ijk->bk", x, y, C)
    got = sparse_gp(x, y, ia, ib, ic, sign)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-5)


def _sort_table(ia, ib, ic, sign):
    # Sort entries by (ia, ib) so independently-constructed tables can be compared.
    keys = ia.to(torch.int64) * (1 << 32) + ib.to(torch.int64)
    order = torch.argsort(keys)
    return ia[order], ib[order], ic[order], sign[order]


def _check_constructors_agree(p: int, q: int, r: int = 0) -> None:
    C = cayley_dense(p, q, r)
    dense = _sort_table(*dense_to_sparse_cayley(C))
    direct = _sort_table(*sparse_cayley_from_sig(p, q, r))
    for a, b in zip(dense, direct, strict=False):
        assert torch.equal(a.to(b.dtype), b)


def test_pga():
    _check_signature(3, 0, 1)
    _check_constructors_agree(3, 0, 1)


def test_cga():
    _check_signature(4, 1, 0)
    _check_constructors_agree(4, 1, 0)


def test_sta():
    _check_signature(1, 3, 0)
    _check_constructors_agree(1, 3, 0)


def _check_sparse_only(p: int, q: int, r: int = 0, batch: int = 4, seed: int = 0) -> None:
    # For larger signatures the dense reference einsum is expensive but still
    # tractable on CPU; we keep the batch tiny to stay under a second at N=8.
    torch.manual_seed(seed)
    C = cayley_dense(p, q, r)
    n = C.shape[0]
    ia, ib, ic, sign = sparse_cayley_from_sig(p, q, r)
    x = torch.randn(batch, n)
    y = torch.randn(batch, n)
    expected = torch.einsum("bi,bj,ijk->bk", x, y, C)
    got = sparse_gp(x, y, ia, ib, ic, sign)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-5)


def test_cma():
    # Cl(2,4,0): conformal Minkowski / twistor algebra. 64 blades.
    _check_constructors_agree(2, 4, 0)
    _check_sparse_only(2, 4, 0)


def test_oct():
    # Cl(8,0,0): 256 blades. The direct-from-sig constructor avoids the
    # 256^3 dense allocation that dense_to_sparse_cayley would otherwise pay.
    _check_constructors_agree(8, 0, 0)
    _check_sparse_only(8, 0, 0)
