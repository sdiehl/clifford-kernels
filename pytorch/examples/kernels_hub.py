import torch

try:
    from kernels import get_kernel
except ImportError:
    raise SystemExit("install with: pip install kernels") from None

cayley = get_kernel("sdiehl/cayley-triton")

ia, ib, ic, sign = cayley.sparse_cayley_from_sig(3, 0, 1)
x = torch.randn(8, 16)
y = torch.randn(8, 16)
out = cayley.sparse_gp(x, y, ia, ib, ic, sign)

print(out.shape)
