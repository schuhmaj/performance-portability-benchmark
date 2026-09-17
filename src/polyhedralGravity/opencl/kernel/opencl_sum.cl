__kernel void sum(__global const FloatType16 *input, __global FloatType16 *blockSums, const int size, __local FloatType *partialSums)
{
  uint id = get_global_id(0);
  uint localId = get_local_id(0);
  FloatType16 value = (id < size) ? input[id] : (FloatType16)(0.0);
  for (uint i = 0; i < 10; ++i) partialSums[localId * 10 + i] = value[i];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
    if (localId < stride) {
      for (uint i = 0; i < 10; ++i) partialSums[localId * 10 + i] += partialSums[(localId + stride) * 10 + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (localId == 0) {
    for (uint i = 0; i < 10; ++i) value[i] = partialSums[i];
    blockSums[get_group_id(0)] = value;
  }
}
