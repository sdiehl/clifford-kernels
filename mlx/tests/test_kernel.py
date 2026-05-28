import mlx.core as mx
import numpy as np
import pytest

from cayley import dense_cayley_from_sig, sparse_cayley_from_sig
from cayley import metal as metal_kernel
from cayley import pure as pure_mlx


def _metal_atomic_float_works():
    try:
        x = mx.zeros((1, 2))
        idx = mx.array([0], dtype=mx.int32)
        sign = mx.array([1.0])
        mx.eval(metal_kernel.sparse_gp(x, x, idx, idx, idx, sign))
        return True
    except Exception:
        return False


requires_metal_atomic = pytest.mark.skipif(
    not _metal_atomic_float_works(),
    reason="atomic_fetch_add on float needs Metal 3.1+ (M3 or newer)",
)


def _np_einsum_ref(p, q, r, batch=4, seed=0):
    mx.random.seed(seed)
    ia, ib, ic, sign = sparse_cayley_from_sig(p, q, r)
    C = dense_cayley_from_sig(p, q, r)
    n_blades = 1 << (p + q + r)
    x = mx.random.normal((batch, n_blades))
    y = mx.random.normal((batch, n_blades))
    mx.eval(x, y)
    expected = np.einsum("bi,bj,ijk->bk", np.array(x), np.array(y), np.array(C))
    return x, y, ia, ib, ic, sign, n_blades, expected


@requires_metal_atomic
@pytest.mark.parametrize(("p", "q", "r"), [(3, 0, 1), (1, 3, 0), (2, 4, 0)])
def test_metal_matches_einsum(p, q, r):
    x, y, ia, ib, ic, sign, _, expected = _np_einsum_ref(p, q, r)
    out = metal_kernel.sparse_gp(x, y, ia, ib, ic, sign)
    mx.eval(out)
    assert np.allclose(np.array(out), expected, atol=1e-5)


@pytest.mark.parametrize(("p", "q", "r"), [(3, 0, 1), (1, 3, 0), (2, 4, 0)])
def test_pure_matches_einsum_cpu(p, q, r):
    # Metal's reduced-precision matmul shows up on GPU; check tight tolerance on CPU.
    mx.set_default_device(mx.cpu)
    try:
        x, y, ia, ib, ic, sign, n_blades, expected = _np_einsum_ref(p, q, r)
        out = pure_mlx.sparse_gp(x, y, ia, ib, ic, sign, n_blades)
        mx.eval(out)
        assert np.allclose(np.array(out), expected, atol=1e-5)
    finally:
        mx.set_default_device(mx.gpu)


@requires_metal_atomic
@pytest.mark.parametrize(("p", "q", "r"), [(3, 0, 1), (1, 3, 0)])
def test_metal_and_pure_agree(p, q, r):
    x, y, ia, ib, ic, sign, n_blades, _ = _np_einsum_ref(p, q, r)
    metal_out = metal_kernel.sparse_gp(x, y, ia, ib, ic, sign)
    pure_out = pure_mlx.sparse_gp(x, y, ia, ib, ic, sign, n_blades)
    mx.eval(metal_out, pure_out)
    assert np.allclose(np.array(metal_out), np.array(pure_out), atol=1e-2)
