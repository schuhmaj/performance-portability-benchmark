__kernel void sum(__global const FloatType16 *input, __global FloatType16 *partialSums, const int size)
{
  uint id = get_global_id(0);
  FloatType16 value = (id < size) ? input[id] : (FloatType16)(0.0);

  for (uint i = 0; i<10; ++i) value[i] = work_group_reduce_add(value[i]);
  if (get_local_id(0) == 0) partialSums[get_group_id(0)] = value;
}
