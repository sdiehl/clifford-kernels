import mlx.core as mx

from cayley_mlx import sparse_cayley_from_sig
from cayley_mlx.pure import sparse_gp

ia, ib, ic, sign = sparse_cayley_from_sig(3, 0, 1)
n_blades = 16
gp = mx.compile(lambda x, y: sparse_gp(x, y, ia, ib, ic, sign))

x = mx.random.normal((4, n_blades))
y = mx.random.normal((4, n_blades))
out = gp(x, y)
mx.eval(out)

print(out.shape)
print(out[0])
