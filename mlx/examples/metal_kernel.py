import mlx.core as mx

from cayley import sparse_cayley_from_sig
from cayley.metal import sparse_gp

ia, ib, ic, sign = sparse_cayley_from_sig(3, 0, 1)
x = mx.random.normal((4, 16))
y = mx.random.normal((4, 16))
out = sparse_gp(x, y, ia, ib, ic, sign)
mx.eval(out)

print(out.shape)
print(out[0])
