import torch

from cayley import dense_to_sparse_cayley, sparse_gp


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


def test_pga():
    _check_signature(3, 0, 1)


def test_cga():
    _check_signature(4, 1, 0)


def test_sta():
    _check_signature(1, 3, 0)
