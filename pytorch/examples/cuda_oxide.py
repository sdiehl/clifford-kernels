# experimental: load the cuda-oxide PTX and launch via cuda-python on torch tensors.
# requires: pip install cuda-python; cd ../rust && cargo oxide build
import ctypes
import re
from pathlib import Path

import torch
from cuda import cuda

from cayley import sparse_cayley_from_sig

ptx = next(
    (Path(__file__).resolve().parent.parent.parent / "rust/target").glob("*/cayley-oxide.ptx")
).read_bytes()
entry = re.search(rb"\.entry\s+(\S+sparse_gp\S*)", ptx).group(1)

torch.zeros(1, device="cuda")  # ensure torch's CUDA context is current
_, module = cuda.cuModuleLoadData(ptx + b"\0")
_, func = cuda.cuModuleGetFunction(module, entry)

ia, ib, ic, sign = (t.cuda() for t in sparse_cayley_from_sig(3, 0, 1))
batch, n_blades = 4, 8
x = torch.randn(batch, n_blades, device="cuda")
y = torch.randn(batch, n_blades, device="cuda")
out = torch.zeros_like(x)

# cuda-oxide ABI: each `&[T]` slice is (ptr u64, len u64); then n_blades u32, batch u32.
vals = []
for t in [x, y, out, ia, ib, ic, sign]:
    vals += [ctypes.c_uint64(t.data_ptr()), ctypes.c_uint64(t.numel())]
vals += [ctypes.c_uint32(n_blades), ctypes.c_uint32(batch)]
ptrs = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])

block = 64
grid = (batch + block - 1) // block
cuda.cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, 0, ptrs, 0)
torch.cuda.synchronize()

print(out[0])
