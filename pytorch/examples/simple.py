import torch

from cayley import dense_cayley_from_sig, dense_to_sparse_cayley, sparse_gp

C = dense_cayley_from_sig(3, 0, 1)
ia, ib, ic, sign = dense_to_sparse_cayley(C)

x = torch.randn(8, 16)
y = torch.randn(8, 16)
out = sparse_gp(x, y, ia, ib, ic, sign)

print(out.shape)
print(out[0])
