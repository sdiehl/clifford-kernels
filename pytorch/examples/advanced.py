import torch
from torch.utils.benchmark import Timer

from cayley_torch import dense_cayley_from_sig, dense_to_sparse_cayley, sparse_gp

device = "cuda" if torch.cuda.is_available() else "cpu"
batch = 4096


def bench(stmt: str) -> float:
    return Timer(stmt, globals=globals()).blocked_autorange(min_run_time=0.5).median


for p, q, r in [(3, 0, 1), (4, 1, 0), (1, 3, 0)]:
    C = dense_cayley_from_sig(p, q, r).to(device)
    ia, ib, ic, sign = (t.to(device) for t in dense_to_sparse_cayley(C))
    n = C.shape[0]
    nnz = ia.numel()

    x = torch.randn(batch, n, device=device)
    y = torch.randn(batch, n, device=device)

    t_sparse = bench("sparse_gp(x, y, ia, ib, ic, sign)")
    t_dense = bench("torch.einsum('bi,bj,ijk->bk', x, y, C)")

    err = (sparse_gp(x, y, ia, ib, ic, sign) - torch.einsum("bi,bj,ijk->bk", x, y, C)).abs().max()
    density = 100 * nnz / n**3
    print(
        f"Cl({p},{q},{r}) n={n:>3} nnz={nnz:>5}/{n**3:<5} ({density:4.1f}%)  "
        f"sparse={t_sparse * 1e3:6.2f}ms  dense={t_dense * 1e3:6.2f}ms  "
        f"speedup={t_dense / t_sparse:5.2f}x  err={err.item():.1e}"
    )
