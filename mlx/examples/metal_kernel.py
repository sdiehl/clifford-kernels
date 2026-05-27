import mlx.core as mx

from cayley import sparse_cayley_from_sig

_SOURCE = """
    uint b = thread_position_in_grid.x;
    if (b >= batch_size) return;
    uint row = b * n_blades;
    for (uint k = 0; k < n_nz; k++) {
        uint a = ia[k];
        uint bb = ib[k];
        uint cc = ic[k];
        float prod = sign[k] * x[row + a] * y[row + bb];
        atomic_fetch_add_explicit(out + (row + cc), prod, memory_order_relaxed);
    }
"""

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


ia, ib, ic, sign = sparse_cayley_from_sig(3, 0, 1)
x = mx.random.normal((4, 16))
y = mx.random.normal((4, 16))
out = sparse_gp(x, y, ia, ib, ic, sign)
mx.eval(out)

print(out.shape)
print(out[0])
