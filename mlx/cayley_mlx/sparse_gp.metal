// One thread per batch row; each thread owns its output row, so no atomics.
uint row = thread_position_in_grid.x * N;
float acc[N] = {0};
for (uint k = 0; k < ia_shape[0]; k++) {
    acc[ic[k]] += float(sign[k]) * float(x[row + ia[k]]) * float(y[row + ib[k]]);
}
for (uint i = 0; i < N; i++) {
    out[row + i] = acc[i];
}
