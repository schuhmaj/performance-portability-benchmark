__kernel void matrix_multiplication(__global const FloatType* a,
                                   __global const FloatType* b,
                                   __global FloatType* c,
                                   const int M, const int N, const int K) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        FloatType sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += a[row + k * M] * b[k + col * K];
        }
        c[row + col * M] = sum;
    }
}
