#include "symbols/NaN.cuh"

// -1/target probability if target = 1.0, 0.0 otherwise
__global__ void backwardCrossEntropyLossKernel (int batchSize, int numberEntriesPerInstance, int numberIterations, float *predictions, float *targets, float *result) {
    // What's the first entry index within the instance that this thread should operate on?
    int startIndexWithinInstance = blockIdx.y * (blockDim.x * numberIterations) + threadIdx.x * numberIterations;

    // Continue if this index is smaller than the dimension of the instance.
    if(startIndexWithinInstance < numberEntriesPerInstance) {
        // What's the first entry index within the batch that this thread should operate on?
        int startIndexWithinBatch = blockIdx.x * numberEntriesPerInstance + startIndexWithinInstance;

        // Is the instance greater than the current batch size?
        if(blockIdx.x >= batchSize) {
            setToNan(result, startIndexWithinBatch, numberIterations);
        }
        else {
            for(int indexEntry = startIndexWithinBatch; indexEntry < startIndexWithinBatch + numberIterations; indexEntry++) {
                result[indexEntry] = targets[indexEntry] * -(1.0/predictions[indexEntry]);
            }
        }
    }

}