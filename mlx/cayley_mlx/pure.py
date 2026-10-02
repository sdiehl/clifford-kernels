import mlx.core as mx


def sparse_gp(x, y, ia, ib, ic, sign):
    contrib = sign * x[:, ia] * y[:, ib]
    return mx.zeros_like(x).at[:, ic].add(contrib)
