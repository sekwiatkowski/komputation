#include "Sigmoid.cuh"

extern "C"
__global__ void backwardSigmoidKernel (int length, double *forward, double *chain, double *destination)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        destination[index] = backwardSigmoid(forward[index], chain[index]);

    }

}