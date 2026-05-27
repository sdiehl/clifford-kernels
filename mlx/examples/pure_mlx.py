import mlx.core as mx

from cayley import sparse_cayley_from_sig
from cayley.pure import sparse_gp

ia, ib, ic, sign = sparse_cayley_from_sig(3, 0, 1)
n_blades = 16
gp = mx.compile(lambda x, y: sparse_gp(x, y, ia, ib, ic, sign, n_blades))

x = mx.random.normal((4, n_blades))
y = mx.random.normal((4, n_blades))
out = gp(x, y)
mx.eval(out)

print(out.shape)
print(out[0])
