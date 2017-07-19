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

template <int blockSize>
__global__ void forwardNormalizationKernel (int numberCategories, double * input, double * result, double * sums)
{

    /*
        The i-th block is responsible for the probability distribution at step i.
        The j-th thread in a block is responsible for the probability of category j at the current step.
    */
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadId;

    /*
        All thread in a block have access to the shared memory.
        The size of the shared memory is set to (number of threads + number of threads / 2) * size per category.

        The extra space beyond the number of categories is needed to make the parallel reduction work. See below.
    */

    extern __shared__ double sharedData[];

    /*
        In many cases, the number of threads will exceed the number of categories.
    */
    if(threadId < numberCategories) {

        sharedData[threadId] = input[globalId];

    }

    __syncthreads();

    /*
        Now that all entries have been exponentiated and put into the shared memory, the normalized probability distributions have to be computed.

        The initial stride is set to the half of number of threads per block: int stride = blockDim.x / 2

        Suppose there are 40 categories. From this it follows that there will be 64 threads. Hence, the initial stride is set to 32.

        stride = number of threads per block / 2 = 32:
        category  0  1  2 ... 7  8 ... 64
                  +  +  +     +  +      +
                 32 33 34    39 40     96

        For this to work, shared memory has to be available for number of threads + number of threads / 2 elements.
        It is further assumed that entries in the shared memory are initialized to be 0.

        In the example, starting from thread 8, zero will be added to the shared memory.

        Last six reductions:
        #6) from 64 to 32
        #5) from 32 to 16
        #4) from 16 to 8
        #3) from 8 to 4
        #2) from 4 to 2
        #1) from 2 to 1

       These last six reductions are unrolled.

    */

    if(blockSize >= 512) {

        sharedData[threadId] += sharedData[threadId + 256];

        __syncthreads();

    }

    if(blockSize >= 256) {

        sharedData[threadId] += sharedData[threadId + 128];

        __syncthreads();

    }

    if(blockSize >= 128) {

        sharedData[threadId] += sharedData[threadId + 64];

        __syncthreads();

    }

    // All warps in a block except for the first one can now be ignored.
    // The condition threadId < 32 returns true only for the first warp. A local variable name would be "isFirstWrap".
    if (threadId < 32) {

        reduceWarp<blockSize>(sharedData, threadId);

    }

    if(threadId == 0) {

        sums[blockId] = sharedData[0];

    }

    result[globalId] = input[globalId] / sharedData[0];

}