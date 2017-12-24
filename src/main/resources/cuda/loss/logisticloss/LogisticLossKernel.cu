#include "reduction/ProductReduction.cuh"

/*
    step 1: 0.8
    step 2: 0.9
    -logf(0.8) + -logf(0.9) = -logf(0.8 * 0.9)
*/

__inline__ __device__ float probability(float prediction, float target) {
    return target * prediction + (1.0 - target) * (1.0 - prediction);
}

__global__ void logisticLossKernel (int batchSize, int numberColumns, int numberIterations, float* predictions, float* targets, float* results)
{
    extern __shared__ float sharedData[];

    int indexInstance = blockIdx.x;

    if(indexInstance < batchSize) {
        float thisValue = 1.0;

        int startInstance = indexInstance * numberColumns;
        int startNextInstance = startInstance + numberColumns;

        int startWithinBatch = startInstance + threadIdx.x * numberIterations;

        for(int indexEntry = startWithinBatch; indexEntry < startWithinBatch + numberIterations; indexEntry++) {
            thisValue *= indexEntry < startNextInstance ? probability(predictions[indexEntry], targets[indexEntry]) : 1.0;
        }

        int warpId = threadIdx.x / warpSize;
        int laneId = threadIdx.x % warpSize;

        reduceToProduct(thisValue, warpId, laneId, sharedData);

        if(threadIdx.x == 0) {
            results[indexInstance] = -logf(sharedData[0]);
        }
    }
    else {
        if(threadIdx.x == 0) {
            results[indexInstance] = 0.0;
        }
    }

}