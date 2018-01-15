#include "../../cuda.h"

__global__ void addKernel (
    float* A,
    float* B,
    float* C,
    int numberIterations,
    int size) {

    int start = (blockIdx.x * blockDim.x + threadIdx.x) * numberIterations;

    for(int entryIndex = start; entryIndex < min(start + numberIterations, size); entryIndex++) {
        C[entryIndex] = A[entryIndex] + B [entryIndex];
    }

}