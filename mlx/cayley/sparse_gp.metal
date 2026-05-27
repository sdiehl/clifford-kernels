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
