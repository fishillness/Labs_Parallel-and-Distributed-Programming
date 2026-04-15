#define __SM_32_INTRINSICS_HPP__

extern "C" __global__ void multiply(float* a, float* b, float* result)
{
    *result = (*a) * (*b);
}