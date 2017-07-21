#include "reduction/Reduction.cuh"

/*
    Example:

                  0.2 0.7
    predictions = 0.3 0.1
                  0.5 0.2

    targets = 0.0 1.0
              1.0 0.0
              0.0 0.0

    number of threads per column = the smallest number equal to the number of rows/categories or greater than the number of rows/categories that is a power of 2 = 4
    number of blocks = number of columns/steps

    shared data in the first block:
    [ 0 * 0.2, 1 * 0.3, 0 * 0.5, 0 * 0.0 ]
    = [ 0, 0.3, 0.0, 0.0 ]

    shared data in the second block:
    [ 1.0 * 0.7, 0.0 * 0.1, 0.0 * 0.2, 0 * 0.0 ]
    = [ 0.7, 0.0, 0.0, 0.0 ]

    parallel sum reduction in the first block:
    [ 0.3, 0.3, 0.0, 0.0 ]

    parallel sum reduction in the second block:
    [ 0.7, 0.0, 0.0, 0.0 ]

    parallel product reduction of the sums of each block:
    0.3 * 0.7 = 0.21

    Negative log:
    -log(0.21) = 0.677780705

*/

template <int blockSize>
__global__ void logisticLossKernel (double *predictions, double *targets, double *sums, double *result)
{

    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int numberBlocks = blockDim.x;

    int globalId = blockId * numberBlocks + threadId;

    extern __shared__ double sharedData[];

    sharedData[threadId] = targets[globalId] * predictions[globalId];

    __syncthreads();

    reduce<blockSize>(threadId, sharedData, 0);

    if(threadId == 0) {

        sums[blockId] = sharedData[0];

    }

    __syncthreads();

    if(globalId == 0) {

        double loss = 1.0;

        for(int index = 0; index < gridDim.x; index++) {

            loss *= sums[index];

        }

        result[0] = -log(loss);

    }

}