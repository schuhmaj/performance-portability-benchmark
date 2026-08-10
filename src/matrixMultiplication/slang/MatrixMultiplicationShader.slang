#ifndef FloatType
#define FloatType double
#define FloatType3 double3
#define FloatType4 double4
#define FloatTypeM dmat3
#endif

struct PushConsts {
    uint M;
    uint N;
    uint K;
};

[[vk::push_constant]]
ConstantBuffer<PushConsts> pc;

[[vk::binding(0, 0)]] StructuredBuffer<FloatType> A;
[[vk::binding(1, 0)]] StructuredBuffer<FloatType> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<FloatType> C;

[shader("compute")]
[numthreads(16, 16, 1)]
void matmul_kernel(uint3 global_id : SV_DispatchThreadID) {
    uint row = global_id.x;
    uint col = global_id.y;

    if (row >= pc.M || col >= pc.N) {
        return;
    }

    FloatType acc = (FloatType)0.0;
    for (uint k = 0; k < pc.K; ++k) {
        acc += A[row + k * pc.M] * B[k + col * pc.K];
    }
    C[row + col * pc.M] = acc;
}
