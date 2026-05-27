import time

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


device = "cuda" if torch.cuda.is_available() else "cpu"
batch, iters = 4096, 20

for p, q, r in [(3, 0, 1), (4, 1, 0), (1, 3, 0)]:
    C = build_cayley(p, q, r).to(device)
    ia, ib, ic, sign = (t.to(device) for t in dense_to_sparse_cayley(C))
    n = C.shape[0]
    nnz = ia.numel()

    x = torch.randn(batch, n, device=device)
    y = torch.randn(batch, n, device=device)

    for _ in range(3):
        sparse_gp(x, y, ia, ib, ic, sign)
        torch.einsum("bi,bj,ijk->bk", x, y, C)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        out_s = sparse_gp(x, y, ia, ib, ic, sign)
    if device == "cuda":
        torch.cuda.synchronize()
    t_sparse = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        out_d = torch.einsum("bi,bj,ijk->bk", x, y, C)
    if device == "cuda":
        torch.cuda.synchronize()
    t_dense = (time.perf_counter() - t0) / iters

    err = (out_s - out_d).abs().max().item()
    density = 100 * nnz / n**3
    print(
        f"Cl({p},{q},{r}) n={n:>3} nnz={nnz:>5}/{n**3:<5} ({density:4.1f}%)  "
        f"sparse={t_sparse * 1e3:6.2f}ms  dense={t_dense * 1e3:6.2f}ms  "
        f"speedup={t_dense / t_sparse:5.2f}x  err={err:.1e}"
    )
