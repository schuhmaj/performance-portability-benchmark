#ifndef FloatType
#define FloatType double
#define FloatType3 double3
#define FloatType4 double4
#define FloatTypeM dmat3
#endif

[[vk::binding(0, 0)]] StructuredBuffer<FloatType4> in_a;
[[vk::binding(1, 0)]] StructuredBuffer<FloatType4> in_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<FloatType4> out_c;

[shader("compute")]
[numthreads(256, 1, 1)]
void vecadd_kernel(uint3 threadId: SV_DispatchThreadID)
{
    uint index = threadId.x;

    uint count, stride;
    in_a.GetDimensions(count, stride);

    if (index < count)
    {
        out_c[index] = in_a[index] + in_b[index];
    }
}