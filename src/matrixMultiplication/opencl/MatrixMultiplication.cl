__kernel void matrix_multiplication_float(__global const float* a,
                                   __global const float* b,
                                   __global float* c,
                                   const int M, const int N, const int K) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += a[row + k * M] * b[k + col * K];
        }
        c[row + col * M] = sum;
    }
}

__kernel void matrix_multiplication_double(__global const double* a,
                                   __global const double* b,
                                   __global double* c,
                                   const int M, const int N, const int K) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        double sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += a[row + k * M] * b[k + col * K];
        }
        c[row + col * M] = sum;
    }
}
