import torch

from cayley import sparse_cayley_from_sig, sparse_gp

ia, ib, ic, sign = sparse_cayley_from_sig(3, 0, 1)
gp = torch.compile(sparse_gp, fullgraph=True)

x = torch.randn(8, 16, requires_grad=True)
y = torch.randn(8, 16, requires_grad=True)

out = gp(x, y, ia, ib, ic, sign)
out.sum().backward()

print(out.shape)
print(x.grad.shape)
