# cayley-oxide

The sparse Cayley table contraction kernel written with [cuda-oxide](https://nvlabs.github.io/cuda-oxide/). The `#[kernel] fn sparse_gp` body in `src/main.rs` translates the Triton kernel into Rust.
cuda-oxide is Linux only and needs an sm_80+ GPU to launch kernels.

Prerequisites: CUDA 12+, LLVM 21+, Clang 21+, and the pinned nightly toolchain (`rust-toolchain.toml` handles that automatically). See the [cuda-oxide installation guide](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html) or use its [devcontainer](https://github.com/NVlabs/cuda-oxide/tree/main/.devcontainer).

Once the toolchain is in place:

```bash
cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
cd rust
cargo oxide build           # compiles the kernel to PTX and embeds it
cargo oxide run             # builds, runs, prints max error vs CPU reference
cargo test                  # CPU-only sig/reference parity tests
```
