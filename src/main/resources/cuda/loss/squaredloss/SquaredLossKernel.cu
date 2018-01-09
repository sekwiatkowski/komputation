#include "reduction/SumReduction.cuh"

// This assumes that the number of threads is equal to the number of predictions/targets.
// First, the squared differences between predictions and targets are stored in shared memory.
// In the second step, the squared differences are summed up using a parallel reduction.
// Finally, the sum is multiplied by 1/2.
__global__ void squaredLossKernel (int batchSize, int numberRows, int numberEntriesPerInstance, int numberIterations, float* predictions, float* targets, float* result) {

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
            thisValue += powf(predictions[startIndexWithinBatch] - targets[startIndexWithinBatch], 2.0);

            if(numberIterations > 1) {
                for(int indexEntry = startIndexWithinBatch + 1; indexEntry < startIndexWithinBatch + numberIterations; indexEntry++) {
                    thisValue += powf(predictions[indexEntry] - targets[indexEntry], 2.0);
                }
            }
        }

        int warpId = threadIdx.x / warpSize;
        int laneId = threadIdx.x % warpSize;

        reduceWarpsToSums(thisValue, warpId, laneId, sharedData);

        if(threadIdx.x == 0) {
            result[indexColumnInBatch] = 0.5 * sharedData[0];
        }
    }
    else {
        if(threadIdx.x == 0) {
            result[indexColumnInBatch] = 0.0;
        }
    }

}