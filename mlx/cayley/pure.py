import mlx.core as mx


def sparse_gp(x, y, ia, ib, ic, sign, n_blades):
    # MLX has no scatter_add; one-hot matmul fuses cleanly under mx.compile.
    contrib = sign * x[:, ia] * y[:, ib]
    one_hot = (ic[:, None] == mx.arange(n_blades)[None, :]).astype(x.dtype)
    return contrib @ one_hot
