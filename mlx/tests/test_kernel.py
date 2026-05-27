import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from cayley import dense_cayley_from_sig, sparse_cayley_from_sig

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pure_mlx = _load("pure_mlx")
metal_kernel = _load("metal_kernel")


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


@pytest.mark.parametrize(("p", "q", "r"), [(3, 0, 1), (1, 3, 0), (2, 4, 0)])
def test_metal_matches_einsum(p, q, r):
    # Metal kernel runs in plain float32 on GPU; exact vs numpy float32 einsum.
    x, y, ia, ib, ic, sign, _, expected = _np_einsum_ref(p, q, r)
    out = metal_kernel.sparse_gp(x, y, ia, ib, ic, sign)
    mx.eval(out)
    assert np.allclose(np.array(out), expected, atol=1e-5)


@pytest.mark.parametrize(("p", "q", "r"), [(3, 0, 1), (1, 3, 0), (2, 4, 0)])
def test_pure_matches_einsum_cpu(p, q, r):
    # Pure-MLX path on the CPU device is exact f32; on GPU Metal's reduced-
    # precision matmul shows up. Run this check on CPU for a tight tolerance.
    mx.set_default_device(mx.cpu)
    try:
        x, y, ia, ib, ic, sign, n_blades, expected = _np_einsum_ref(p, q, r)
        out = pure_mlx.sparse_gp(x, y, ia, ib, ic, sign, n_blades)
        mx.eval(out)
        assert np.allclose(np.array(out), expected, atol=1e-5)
    finally:
        mx.set_default_device(mx.gpu)


@pytest.mark.parametrize(("p", "q", "r"), [(3, 0, 1), (1, 3, 0)])
def test_metal_and_pure_agree(p, q, r):
    # Both implementations on the default device should produce results within
    # Metal's matmul precision of each other.
    x, y, ia, ib, ic, sign, n_blades, _ = _np_einsum_ref(p, q, r)
    metal_out = metal_kernel.sparse_gp(x, y, ia, ib, ic, sign)
    pure_out = pure_mlx.sparse_gp(x, y, ia, ib, ic, sign, n_blades)
    mx.eval(metal_out, pure_out)
    assert np.allclose(np.array(metal_out), np.array(pure_out), atol=1e-2)
