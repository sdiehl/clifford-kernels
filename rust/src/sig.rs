pub struct Cayley {
    pub ia: Vec<i32>,
    pub ib: Vec<i32>,
    pub ic: Vec<i32>,
    pub sign: Vec<f32>,
}

fn popcount(mut n: u32) -> u32 {
    let mut c = 0;
    while n != 0 {
        n &= n - 1;
        c += 1;
    }
    c
}

fn swap_sign(a: u32, b: u32, dim: u32) -> i32 {
    let mut count = 0u32;
    for j in 0..dim {
        let above = a >> (j + 1);
        if (b >> j) & 1 == 1 {
            let mask = (1u32 << (dim - j - 1)) - 1;
            count += popcount(above & mask);
        }
    }
    if count.is_multiple_of(2) { 1 } else { -1 }
}

fn basis_sign(i: u32, p: u32, q: u32) -> i32 {
    if i < p {
        1
    } else if i < p + q {
        -1
    } else {
        0
    }
}

fn metric_sign(mask: u32, p: u32, q: u32, dim: u32) -> i32 {
    let mut acc = 1i32;
    for i in 0..dim {
        if (mask >> i) & 1 == 1 {
            let m = basis_sign(i, p, q);
            if m == 0 {
                return 0;
            }
            acc *= m;
        }
    }
    acc
}

pub fn sparse_cayley_from_sig(p: u32, q: u32, r: u32) -> Cayley {
    let dim = p + q + r;
    let n = 1u32 << dim;
    let mut ia = Vec::new();
    let mut ib = Vec::new();
    let mut ic = Vec::new();
    let mut sign = Vec::new();
    for a in 0..n {
        for b in 0..n {
            let m = metric_sign(a & b, p, q, dim);
            if m == 0 {
                continue;
            }
            let s = swap_sign(a, b, dim) * m;
            ia.push(a as i32);
            ib.push(b as i32);
            ic.push((a ^ b) as i32);
            sign.push(s as f32);
        }
    }
    Cayley { ia, ib, ic, sign }
}
