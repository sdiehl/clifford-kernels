use crate::sig::Cayley;

pub fn sparse_gp_cpu(x: &[f32], y: &[f32], batch: usize, n_blades: usize, c: &Cayley) -> Vec<f32> {
    sparse_gp_cpu_indexed(x, y, batch, n_blades, &c.ia, &c.ib, &c.ic, &c.sign)
}

#[allow(clippy::too_many_arguments)]
pub fn sparse_gp_cpu_indexed(
    x: &[f32],
    y: &[f32],
    batch: usize,
    n_blades: usize,
    ia: &[i32],
    ib: &[i32],
    ic: &[i32],
    sign: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * n_blades];
    for b in 0..batch {
        let row = b * n_blades;
        for k in 0..ia.len() {
            let a = ia[k] as usize;
            let bb = ib[k] as usize;
            let cc = ic[k] as usize;
            out[row + cc] += sign[k] * x[row + a] * y[row + bb];
        }
    }
    out
}
