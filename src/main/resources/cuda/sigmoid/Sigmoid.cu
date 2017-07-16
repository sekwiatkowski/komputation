#include "Sigmoid.cuh"

extern "C"
__global__ void sigmoidKernel (int length, double *source, double *destination)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        destination[index] = sigmoid(source[index]);

    }

}