import mlx.core as mx
import numpy as np
import pytest

from cayley_mlx import dense_cayley_from_sig, sparse_cayley_from_sig
from cayley_mlx import metal as metal_kernel
from cayley_mlx import pure as pure_mlx

SIGS = [(3, 0, 1), (1, 3, 0), (2, 4, 0)]


def _np_einsum_ref(p, q, r, batch=4, seed=0):
    mx.random.seed(seed)
    ia, ib, ic, sign = sparse_cayley_from_sig(p, q, r)
    C = dense_cayley_from_sig(p, q, r)
    n_blades = 1 << (p + q + r)
    x = mx.random.normal((batch, n_blades))
    y = mx.random.normal((batch, n_blades))
    mx.eval(x, y)
    expected = np.einsum("bi,bj,ijk->bk", np.array(x), np.array(y), np.array(C))
    return x, y, ia, ib, ic, sign, expected


@pytest.mark.parametrize("impl", [metal_kernel, pure_mlx], ids=["metal", "pure"])
@pytest.mark.parametrize(("p", "q", "r"), SIGS)
def test_matches_einsum(impl, p, q, r):
    x, y, ia, ib, ic, sign, expected = _np_einsum_ref(p, q, r)
    out = impl.sparse_gp(x, y, ia, ib, ic, sign)
    assert np.allclose(np.array(out), expected, atol=1e-5)


@pytest.mark.parametrize(("p", "q", "r"), SIGS)
def test_metal_vjp_matches_pure(p, q, r):
    x, y, ia, ib, ic, sign, _ = _np_einsum_ref(p, q, r)
    w = mx.random.normal(x.shape)

    def loss(impl):
        return lambda x, y: (impl.sparse_gp(x, y, ia, ib, ic, sign) * w).sum()

    got = mx.grad(loss(metal_kernel), argnums=(0, 1))(x, y)
    expected = mx.grad(loss(pure_mlx), argnums=(0, 1))(x, y)
    for g, e in zip(got, expected, strict=True):
        assert np.allclose(np.array(g), np.array(e), atol=1e-4)
