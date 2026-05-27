from importlib.resources import files

import mlx.core as mx

_SOURCE = (files(__package__) / "sparse_gp.metal").read_text()

_kernel = mx.fast.metal_kernel(
    name="sparse_gp",
    input_names=["x", "y", "ia", "ib", "ic", "sign"],
    output_names=["out"],
    source=_SOURCE,
    atomic_outputs=True,
)


def sparse_gp(x, y, ia, ib, ic, sign):
    batch, n_blades = x.shape
    n_nz = ia.shape[0]
    (out,) = _kernel(
        inputs=[x, y, ia, ib, ic, sign],
        template=[("n_nz", n_nz), ("n_blades", n_blades), ("batch_size", batch)],
        grid=(batch, 1, 1),
        threadgroup=(min(batch, 64), 1, 1),
        output_shapes=[(batch, n_blades)],
        output_dtypes=[x.dtype],
        init_value=0.0,
    )
    return out
