import torch

from cayley import dense_to_sparse_cayley, sparse_gp


def build_cayley(p, q, r):
    d = p + q + r
    n = 1 << d
    C = torch.zeros(n, n, n)
    for a in range(n):
        for b in range(n):
            s, keep = 1, True
            for j in range(d):
                if (b >> j) & 1:
                    above = (a >> (j + 1)) & ((1 << (d - j - 1)) - 1)
                    if bin(above).count("1") & 1:
                        s = -s
            for i in range(d):
                if (a >> i) & 1 and (b >> i) & 1:
                    if i >= p + q:
                        keep = False
                        break
                    if i >= p:
                        s = -s
            if keep:
                C[a, b, a ^ b] = s
    return C


C = build_cayley(3, 0, 1)
ia, ib, ic, sign = dense_to_sparse_cayley(C)

x = torch.randn(8, 16)
y = torch.randn(8, 16)
out = sparse_gp(x, y, ia, ib, ic, sign)

print(out.shape)
print(out[0])
