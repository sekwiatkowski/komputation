__inline__ __device__ float relu (float x)
{
    return fmaxf(x, 0.0);
}