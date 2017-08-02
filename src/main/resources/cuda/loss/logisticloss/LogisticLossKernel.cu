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
__global__ void logisticLossKernel (int batchSize, int numberRows, int numberEntriesPerInstance, int numberIterations, float *predictions, float *targets, float *result)
{

    int startIndexWithinColumn = threadIdx.x * numberIterations;

    extern __shared__ float sharedData[];

    int indexInstance = blockIdx.x;
    int indexColumn = blockIdx.y;

    int startIndexWithinInstance = indexColumn * numberRows + startIndexWithinColumn;
    int startIndexWithinBatch = indexInstance * numberEntriesPerInstance + startIndexWithinInstance;

    int indexColumnInBatch = indexInstance * gridDim.y + indexColumn;

    if(indexInstance < batchSize) {

        if(startIndexWithinColumn < numberRows) {

            float sum = 0.0;

            for(int indexEntry = startIndexWithinBatch; indexEntry < startIndexWithinBatch + numberIterations; indexEntry++) {

                sum += targets[indexEntry] * predictions[indexEntry];

            }

            sharedData[threadIdx.x] = sum;

        }
        else {

            sharedData[threadIdx.x] = 0.0;

        }

        __syncthreads();

        reduce<blockSize>(threadIdx.x, sharedData, 0);

        if(threadIdx.x == 0) {

            result[indexColumnInBatch] = -logf(sharedData[0]);

        }

    }
    else {

        if(threadIdx.x == 0) {

            result[indexColumnInBatch] = 0.0;

        }

    }

}