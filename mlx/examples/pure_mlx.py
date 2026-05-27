import mlx.core as mx

from cayley import sparse_cayley_from_sig


def sparse_gp(x, y, ia, ib, ic, sign, n_blades):
    # Gather contributions for each (k, batch), then sum into out[:, ic[k]]
    # via a one-hot matmul. MLX has no scatter_add primitive; the one-hot trick
    # is the idiomatic stand-in and fuses cleanly under mx.compile.
    contrib = sign * x[:, ia] * y[:, ib]
    one_hot = (ic[:, None] == mx.arange(n_blades)[None, :]).astype(x.dtype)
    return contrib @ one_hot


ia, ib, ic, sign = sparse_cayley_from_sig(3, 0, 1)
n_blades = 16
gp = mx.compile(lambda x, y: sparse_gp(x, y, ia, ib, ic, sign, n_blades))

x = mx.random.normal((4, n_blades))
y = mx.random.normal((4, n_blades))
out = gp(x, y)
mx.eval(out)

print(out.shape)
print(out[0])
