from collections.abc import Iterator

import mlx.core as mx


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


def _cayley_entries(p: int, q: int, r: int) -> Iterator[tuple[int, int, int, int]]:
    dim = p + q + r
    n = 1 << dim
    for a in range(n):
        for b in range(n):
            m = _metric_sign(a & b, p, q, dim)
            if m == 0:
                continue
            yield a, b, a ^ b, _swap_sign(a, b, dim) * m


def sparse_cayley_from_sig(
    p: int, q: int, r: int = 0, *, dtype: mx.Dtype = mx.float32
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    ia_l, ib_l, ic_l, sgn_l = [], [], [], []
    for a, b, c, s in _cayley_entries(p, q, r):
        ia_l.append(a)
        ib_l.append(b)
        ic_l.append(c)
        sgn_l.append(s)
    return (
        mx.array(ia_l, dtype=mx.int32),
        mx.array(ib_l, dtype=mx.int32),
        mx.array(ic_l, dtype=mx.int32),
        mx.array(sgn_l, dtype=dtype),
    )


def dense_cayley_from_sig(
    p: int, q: int, r: int = 0, *, dtype: mx.Dtype = mx.float32
) -> mx.array:
    n = 1 << (p + q + r)
    arr = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for a, b, c, s in _cayley_entries(p, q, r):
        arr[a][b][c] = s
    return mx.array(arr, dtype=dtype)
