#include "../../cuda.h"
#include "../../symbols/NaN.cuh"

__global__ void backwardLogisticLossKernel (int batchSize, int numberColumns, int numberIterations, float* predictions, float* targets, float* results) {
    int indexInstance = blockIdx.x;

    int startInstanceWithinBatch = indexInstance * numberColumns;
    int startNextInstanceWithinBatch = startInstanceWithinBatch + numberColumns;

    if(indexInstance < batchSize) {
        for(int indexEntry = startInstanceWithinBatch + threadIdx.x * numberIterations; indexEntry < startNextInstanceWithinBatch; indexEntry++) {

            float target = targets[indexEntry];
            float prediction = predictions[indexEntry];

            float positive = -1.0f/prediction;
            float negative = 1.0f/(1.0f - prediction);

            results[indexEntry] = target * positive + (1.0f - target) * negative;
        }
    }
    else {
        setToNaN(results, startInstanceWithinBatch, startNextInstanceWithinBatch);
    }
}