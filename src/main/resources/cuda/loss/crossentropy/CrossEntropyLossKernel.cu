#include "reduction/SumReduction.cuh"

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

__global__ void crossEntropyLossKernel (int batchSize, int numberRows, int numberEntriesPerInstance, int numberIterations, float* predictions, float* targets, float* result)
{

    int startIndexWithinColumn = threadIdx.x * numberIterations;

    extern __shared__ float sharedData[];

    int indexInstance = blockIdx.x;
    int indexColumn = blockIdx.y;

    int startIndexWithinInstance = indexColumn * numberRows + startIndexWithinColumn;
    int startIndexWithinBatch = indexInstance * numberEntriesPerInstance + startIndexWithinInstance;

    int indexColumnInBatch = indexInstance * gridDim.y + indexColumn;

    if(indexInstance < batchSize) {
        float thisValue = 0.0;

        if(startIndexWithinColumn < numberRows) {
            thisValue = targets[startIndexWithinBatch] * predictions[startIndexWithinBatch];

            if(numberIterations > 1) {
                for(int indexEntry = startIndexWithinBatch + 1; indexEntry < startIndexWithinBatch + numberIterations; indexEntry++) {
                    thisValue += targets[indexEntry] * predictions[indexEntry];
                }
            }
        }

        int warpId = threadIdx.x / warpSize;
        int laneId = threadIdx.x % warpSize;

        reduceToSum(thisValue, warpId, laneId, sharedData);

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