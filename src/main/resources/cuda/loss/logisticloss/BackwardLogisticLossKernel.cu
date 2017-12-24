#include "symbols/NaN.cuh"

__global__ void backwardLogisticLossKernel (int batchSize, int numberColumns, int numberIterations, float* predictions, float* targets, float* results) {
    int indexInstance = blockIdx.x;

    int startInstanceWithinBatch = indexInstance * numberColumns;
    int startNextInstanceWithinBatch = startInstanceWithinBatch + numberColumns;

    if(indexInstance < batchSize) {
        for(int indexEntry = startInstanceWithinBatch + threadIdx.x * numberIterations; indexEntry < startNextInstanceWithinBatch; indexEntry++) {

            float target = targets[indexEntry];
            float prediction = predictions[indexEntry];

            float positive = -1.0/prediction;
            float negative = 1.0/(1.0 - prediction);

            results[indexEntry] = target * positive + (1.0 - target) * negative;
        }
    }
    else {
        setToNan(results, startInstanceWithinBatch, startNextInstanceWithinBatch);
    }
}