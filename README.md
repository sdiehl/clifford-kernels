# clifford-kernels

Sparse [Cayley table](https://en.wikipedia.org/wiki/Cayley_table) contraction kernels for the geometric product in Clifford algebras. The Cayley tensor of $`\mathrm{Cl}(p,q,r)`$ is ~90% zeros, so a dense Einstein summation wastes most of its FLOPs. Both implementations extract the nonzeros as `(ia, ib, ic, sign)` and walk only those, with an atomic-add per nonzero into the output.

This class of transformers are useful in some applications where the underlying training data exists uniformly on some higher dimensional manifold with particular types of physical symmetries encoded by the algebra. This shows up in molecular force prediction, HEP top quark tagging, and robotics applications. These kernels are designed to be used as building blocks in larger models.

Three implementations of the same kernel:

- [**pytorch**](pytorch/) — a Triton kernel, differentiable through `torch.autograd`, with a CPU fallback
- [**cuda-oxide**](rust/) — the same kernel written in pure Rust using [cuda-oxide](https://nvlabs.github.io/cuda-oxide/) NVidia's Rust-to-PTX compiler
- [**mlx**](mlx/) — Same but for Apple Silicon using a `mlx.compile` or a custom Metal shader kernel via `mx.fast.metal_kernel`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
