use cayley_oxide::sig::sparse_cayley_from_sig;

#[test]
fn cl_3_0_0_sizes_match_python() {
    let c = sparse_cayley_from_sig(3, 0, 0);
    assert_eq!(c.ia.len(), c.ib.len());
    assert_eq!(c.ib.len(), c.ic.len());
    assert_eq!(c.ic.len(), c.sign.len());
    let n: u32 = 1 << 3;
    assert!(c.ia.iter().all(|&v| (v as u32) < n));
    assert!(c.ib.iter().all(|&v| (v as u32) < n));
    assert!(c.ic.iter().all(|&v| (v as u32) < n));
    assert!(c.sign.iter().all(|&s| s == 1.0 || s == -1.0));
}

#[test]
fn cl_3_0_0_geometric_product_identity() {
    let c = sparse_cayley_from_sig(3, 0, 0);
    let n_blades = 8usize;
    let mut x = vec![0.0f32; n_blades];
    let mut y = vec![0.0f32; n_blades];
    x[0] = 1.0;
    y[3] = 2.5;
    let out = cayley_oxide::reference::sparse_gp_cpu(&x, &y, 1, n_blades, &c);
    assert_eq!(out[3], 2.5);
    for (i, v) in out.iter().enumerate().take(n_blades) {
        if i != 3 {
            assert_eq!(*v, 0.0);
        }
    }
}

#[test]
fn cl_2_0_0_e1_squared_is_one() {
    let c = sparse_cayley_from_sig(2, 0, 0);
    let n_blades = 4usize;
    let mut x = vec![0.0f32; n_blades];
    x[1] = 1.0;
    let out = cayley_oxide::reference::sparse_gp_cpu(&x, &x, 1, n_blades, &c);
    assert_eq!(out[0], 1.0);
}

#[test]
fn cl_0_1_0_e1_squared_is_minus_one() {
    let c = sparse_cayley_from_sig(0, 1, 0);
    let n_blades = 2usize;
    let mut x = vec![0.0f32; n_blades];
    x[1] = 1.0;
    let out = cayley_oxide::reference::sparse_gp_cpu(&x, &x, 1, n_blades, &c);
    assert_eq!(out[0], -1.0);
}
