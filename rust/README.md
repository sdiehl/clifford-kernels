# cayley-oxide

The sparse Cayley table contraction kernel from the root [README](../README.md), written in pure Rust using [cuda-oxide](https://nvlabs.github.io/cuda-oxide/). The `#[kernel] fn sparse_gp` body in `src/main.rs` translates the Triton kernel in [`../pytorch/cayley/kernel.py`](../pytorch/cayley/kernel.py)

The driver in `main()` reproduces `../pytorch/examples/simple.py`: it builds the Cayley table for `Cl(3,0,0)` from `src/sig.rs` (a port of `cayley/sig.py`), runs the kernel, then runs the two extra launches with permuted indices that compute `dx` and `dy`.

cuda-oxide is **Linux only** and needs an Ampere+ NVIDIA GPU (sm_80+) to launch kernels. Prerequisites: CUDA 12+, LLVM 21+, Clang 21+, and the pinned nightly toolchain (`rust-toolchain.toml` handles that automatically). See the [cuda-oxide installation guide](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html) or use its [devcontainer](https://github.com/NVlabs/cuda-oxide/tree/main/.devcontainer).

Once the toolchain is in place:

```bash
cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
cd rust
cargo oxide build           # compiles the kernel to PTX and embeds it
cargo oxide run             # builds, runs, prints max error vs CPU reference
cargo test                  # CPU-only sig/reference parity tests
```
