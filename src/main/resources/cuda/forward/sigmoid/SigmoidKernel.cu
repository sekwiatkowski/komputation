#include "zero/Zero.cuh"

__device__ float sigmoid (float x) {

    return 1.0 / (1.0 + expf (-x));

}

extern "C"
__global__ void sigmoidKernel (
    int batchSize,
    int numberEntriesPerInstance,
    int numberIterations,
    float *source,
    float *destination) {

    int indexInstance = blockIdx.x;

    int startInstanceWithinBatch = indexInstance * numberEntriesPerInstance;
    int startNextInstanceWithinBatch = startInstanceWithinBatch + numberEntriesPerInstance;

    int firstEntryWithinBatch = startInstanceWithinBatch + blockIdx.y * blockDim.x * numberIterations + threadIdx.x * numberIterations;

    if(firstEntryWithinBatch < startNextInstanceWithinBatch) {

        int lastEntryWithinBatch = min(firstEntryWithinBatch + numberIterations, startNextInstanceWithinBatch);

        if(indexInstance < batchSize) {

            for(int indexEntry = firstEntryWithinBatch; indexEntry < lastEntryWithinBatch; indexEntry++) {

                destination[indexEntry] = sigmoid(source[indexEntry]);

            }

        }
        else {

            setToZero(destination, firstEntryWithinBatch, lastEntryWithinBatch);

        }

    }

}