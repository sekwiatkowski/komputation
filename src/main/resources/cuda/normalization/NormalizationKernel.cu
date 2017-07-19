#include "reduction/Reduction.cuh"

/*

    The number of categories is assumed to be smaller than or equal to the maximum number of threads per block.
    The warp size is further assumed to be 32.

    number of warps = ceil(number of categories / warp size)
    number of threads per block = number of warps * warp size

    Examples:
    1 category => ceil(1 / 32) * 32 = 32 threads
    31 categories => ceil(31 / 32) * 32 = 32 threads
    32 categories => ceil(32 / 32) * 32 = 32 threads
    33 categories => ceil(33 / 32) * 32 = 64 threads
    64 categories => ceil(64 / 32) * 32 = 64 threads
    65 categories => ceil(65 / 32) * 32 = 96 threads

*/

/*
    The size of the shared memory is set to (number of threads + number of threads / 2) * size per category.

    The extra space beyond the number of categories is needed to make the parallel reduction work. See thereduce function.
*/

/*
    This kernel essentially has two steps:
    (1) The entries in each column are summed up
    (2) Each column entry is divided by that sum.

    The column sums are stored for reuse in the backward propagation kernel.
*/


template <int blockSize>
__global__ void normalizationKernel (int numberCategories, double* input, double* result, double* sums)
{

    /*
        The i-th block is responsible for the probability distribution at step i.
        The j-th thread in a block is responsible for the probability of category j at the current step.
    */
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadId;

    extern __shared__ double sharedData[];

    // Recall that the block size is a power of 2.
    // In many cases the number of threads will thus exceed the number of categories.
    if(threadId < numberCategories) {

        sharedData[threadId] = input[globalId];

    }

    __syncthreads();

    // Computing the sum of entries in each column is a parallel reduction operation.
    reduce<blockSize>(threadId, sharedData, 0);

    if(threadId == 0) {

        sums[blockId] = sharedData[0];

    }

    result[globalId] = input[globalId] / sharedData[0];

}