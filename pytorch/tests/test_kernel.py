import torch

from cayley import (
    dense_cayley_from_sig,
    dense_to_sparse_cayley,
    sparse_cayley_from_sig,
    sparse_gp,
)


def _check_kernel_matches_einsum(
    p: int,
    q: int,
    r: int = 0,
    *,
    use_direct_sparse: bool = False,
    batch: int = 4,
    seed: int = 0,
) -> None:
    # For larger signatures the dense reference einsum is expensive but still
    # tractable on CPU; we keep the batch tiny to stay under a second at N=8.
    torch.manual_seed(seed)
    C = dense_cayley_from_sig(p, q, r)
    n = C.shape[0]
    if use_direct_sparse:
        ia, ib, ic, sign = sparse_cayley_from_sig(p, q, r)
    else:
        ia, ib, ic, sign = dense_to_sparse_cayley(C)
    x = torch.randn(batch, n)
    y = torch.randn(batch, n)
    expected = torch.einsum("bi,bj,ijk->bk", x, y, C)
    got = sparse_gp(x, y, ia, ib, ic, sign)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-5)


def _sort_table(ia, ib, ic, sign):
    # Sort entries by (ia, ib) so independently-constructed tables can be compared.
    keys = ia.to(torch.int64) * (1 << 32) + ib.to(torch.int64)
    order = torch.argsort(keys)
    return ia[order], ib[order], ic[order], sign[order]


def _check_constructors_agree(p: int, q: int, r: int = 0) -> None:
    C = dense_cayley_from_sig(p, q, r)
    dense = _sort_table(*dense_to_sparse_cayley(C))
    direct = _sort_table(*sparse_cayley_from_sig(p, q, r))
    for a, b in zip(dense, direct, strict=False):
        assert torch.equal(a.to(b.dtype), b)


def test_pga():
    _check_kernel_matches_einsum(3, 0, 1)
    _check_constructors_agree(3, 0, 1)


def test_cga():
    _check_kernel_matches_einsum(4, 1, 0)
    _check_constructors_agree(4, 1, 0)


def test_sta():
    _check_kernel_matches_einsum(1, 3, 0)
    _check_constructors_agree(1, 3, 0)


def test_cma():
    # Cl(2,4,0): conformal Minkowski / twistor algebra. 64 blades.
    _check_constructors_agree(2, 4, 0)
    _check_kernel_matches_einsum(2, 4, 0, use_direct_sparse=True)


def test_oct():
    # Cl(8,0,0): 256 blades. The direct-from-sig constructor avoids the
    # 256^3 dense allocation that dense_to_sparse_cayley would otherwise pay.
    _check_constructors_agree(8, 0, 0)
    _check_kernel_matches_einsum(8, 0, 0, use_direct_sparse=True)


def _check_gradient(p: int, q: int, r: int = 0, batch: int = 3, seed: int = 0) -> None:
    torch.manual_seed(seed)
    C = dense_cayley_from_sig(p, q, r, dtype=torch.float64)
    n = C.shape[0]
    ia, ib, ic, sign = sparse_cayley_from_sig(p, q, r, dtype=torch.float64)

    x_ref = torch.randn(batch, n, dtype=torch.float64)
    y_ref = torch.randn(batch, n, dtype=torch.float64)
    dout = torch.randn(batch, n, dtype=torch.float64)

    x_s = x_ref.clone().requires_grad_(True)
    y_s = y_ref.clone().requires_grad_(True)
    out_s = sparse_gp(x_s, y_s, ia, ib, ic, sign)
    (dx_s, dy_s) = torch.autograd.grad(out_s, (x_s, y_s), grad_outputs=dout)

    x_d = x_ref.clone().requires_grad_(True)
    y_d = y_ref.clone().requires_grad_(True)
    out_d = torch.einsum("bi,bj,ijk->bk", x_d, y_d, C)
    (dx_d, dy_d) = torch.autograd.grad(out_d, (x_d, y_d), grad_outputs=dout)

    assert torch.allclose(out_s, out_d, atol=1e-10)
    assert torch.allclose(dx_s, dx_d, atol=1e-10)
    assert torch.allclose(dy_s, dy_d, atol=1e-10)


def test_grad_pga():
    _check_gradient(3, 0, 1)


def test_grad_sta():
    _check_gradient(1, 3, 0)


def test_grad_cma():
    _check_gradient(2, 4, 0)


def test_registered_as_custom_op():
    # The op should be reachable via torch.ops, confirming the
    # torch.library.custom_op registration succeeded.
    assert hasattr(torch.ops, "cayley")
    assert hasattr(torch.ops.cayley, "sparse_gp")
