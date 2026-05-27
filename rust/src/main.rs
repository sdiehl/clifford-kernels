#[cfg(feature = "gpu")]
mod gpu {
    use cayley_oxide::{reference, sig};
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
    use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
    use cuda_device::{cuda_module, kernel, thread};

    #[cuda_module]
    mod kernels {
        use super::*;

        #[kernel]
        pub fn sparse_gp(
            x: &[f32],
            y: &[f32],
            out: &[f32],
            ia: &[i32],
            ib: &[i32],
            ic: &[i32],
            sign: &[f32],
            n_blades: u32,
            batch_size: u32,
        ) {
            let b = thread::index_1d().get();
            if b >= batch_size as usize {
                return;
            }
            let row = b * n_blades as usize;
            let n_nz = ia.len();
            let mut k = 0;
            while k < n_nz {
                let a = ia[k] as usize;
                let bb = ib[k] as usize;
                let cc = ic[k] as usize;
                let s = sign[k];
                let prod = s * x[row + a] * y[row + bb];
                let slot = unsafe { &*(out.as_ptr().add(row + cc) as *const DeviceAtomicF32) };
                slot.fetch_add(prod, AtomicOrdering::Relaxed);
                k += 1;
            }
        }
    }

    pub fn run() {
        let p = 3u32;
        let q = 0u32;
        let r = 0u32;
        let c = sig::sparse_cayley_from_sig(p, q, r);
        let n_blades = 1usize << (p + q + r);
        let batch = 4usize;

        let x_host: Vec<f32> = (0..batch * n_blades).map(|i| (i as f32) * 0.01).collect();
        let y_host: Vec<f32> = (0..batch * n_blades)
            .map(|i| (i as f32) * 0.02 + 1.0)
            .collect();

        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx).expect("load kernel module");

        let x_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();
        let y_dev = DeviceBuffer::from_host(&stream, &y_host).unwrap();
        let out_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * n_blades).unwrap();

        let ia_dev = DeviceBuffer::from_host(&stream, &c.ia).unwrap();
        let ib_dev = DeviceBuffer::from_host(&stream, &c.ib).unwrap();
        let ic_dev = DeviceBuffer::from_host(&stream, &c.ic).unwrap();
        let sign_dev = DeviceBuffer::from_host(&stream, &c.sign).unwrap();

        let cfg = LaunchConfig::for_num_elems(batch as u32);

        module
            .sparse_gp(
                &stream,
                cfg,
                &x_dev,
                &y_dev,
                &out_dev,
                &ia_dev,
                &ib_dev,
                &ic_dev,
                &sign_dev,
                n_blades as u32,
                batch as u32,
            )
            .expect("forward launch");

        let out_host = out_dev.to_host_vec(&stream).unwrap();
        let expected = reference::sparse_gp_cpu(&x_host, &y_host, batch, n_blades, &c);
        let max_err = out_host
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!(
            "Cl({},{},{}) batch={} n_blades={} nnz={}",
            p,
            q,
            r,
            batch,
            n_blades,
            c.ia.len()
        );
        println!("max abs error vs CPU reference: {:.3e}", max_err);
        println!(
            "out[0..{}]: {:?}",
            n_blades.min(8),
            &out_host[..n_blades.min(8)]
        );

        let dx_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * n_blades).unwrap();
        let dy_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * n_blades).unwrap();
        let dout_dev = DeviceBuffer::from_host(&stream, &vec![1.0f32; batch * n_blades]).unwrap();

        module
            .sparse_gp(
                &stream,
                cfg,
                &y_dev,
                &dout_dev,
                &dx_dev,
                &ib_dev,
                &ic_dev,
                &ia_dev,
                &sign_dev,
                n_blades as u32,
                batch as u32,
            )
            .expect("backward dx launch");
        module
            .sparse_gp(
                &stream,
                cfg,
                &x_dev,
                &dout_dev,
                &dy_dev,
                &ia_dev,
                &ic_dev,
                &ib_dev,
                &sign_dev,
                n_blades as u32,
                batch as u32,
            )
            .expect("backward dy launch");

        let _ = (
            dx_dev.to_host_vec(&stream).unwrap(),
            dy_dev.to_host_vec(&stream).unwrap(),
        );
        println!("backward dx/dy launched successfully");
    }
}

#[cfg(feature = "gpu")]
fn main() {
    gpu::run();
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("cayley-oxide binary requires --features gpu (Linux + CUDA only).");
    std::process::exit(1);
}
