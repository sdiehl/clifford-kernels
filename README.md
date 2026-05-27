# cayley-triton

Sparse [Cayley table](https://en.wikipedia.org/wiki/Cayley_table) contraction kernels for the geometric product in Clifford algebras. The Cayley tensor of $`\mathrm{Cl}(p,q,r)`$ is ~90% zeros, so a dense Einstein summation wastes most of its FLOPs. Both implementations extract the nonzeros as `(ia, ib, ic, sign)` and walk only those, with an atomic-add per nonzero into the output.

Two implementations of the same kernel:

- [**pytorch/**](pytorch/) — a Triton kernel, differentiable through `torch.autograd`, with a CPU fallback
- [**rust/**](rust/) — the same kernel written in pure Rust using [cuda-oxide](https://nvlabs.github.io/cuda-oxide/) NVidia's Rust-to-PTX compiler

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
