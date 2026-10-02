from importlib.resources import files

import mlx.core as mx

_SOURCE = (files(__package__) / "sparse_gp.metal").read_text()

_kernel = mx.fast.metal_kernel(
    name="sparse_gp",
    input_names=["x", "y", "ia", "ib", "ic", "sign"],
    output_names=["out"],
    source=_SOURCE,
)


def _launch(x, y, ia, ib, ic, sign):
    batch, n_blades = x.shape
    (out,) = _kernel(
        inputs=[x, y, ia, ib, ic, sign],
        template=[("N", n_blades)],
        grid=(batch, 1, 1),
        threadgroup=(min(batch, 256), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


@mx.custom_function
def sparse_gp(x, y, ia, ib, ic, sign):
    return _launch(x, y, ia, ib, ic, sign)


@sparse_gp.vjp
def _sparse_gp_vjp(primals, cotangent, output):
    x, y, ia, ib, ic, sign = primals
    dx = _launch(y, cotangent, ib, ic, ia, sign)
    dy = _launch(x, cotangent, ia, ic, ib, sign)
    return dx, dy, mx.zeros_like(ia), mx.zeros_like(ib), mx.zeros_like(ic), mx.zeros_like(sign)
