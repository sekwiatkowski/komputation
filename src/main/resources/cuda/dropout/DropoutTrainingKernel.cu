#include <curand>

extern "C"
__global__ void dropoutTrainingKernel (float* result)
{

    result[0] = 1.0;

}