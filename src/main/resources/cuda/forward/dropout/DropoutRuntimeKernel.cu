#include "zero/Zero.cuh"

extern "C"
__global__ void dropoutRuntimeKernel (
    int batchSize,
    int numberEntriesPerInstance,
    int numberIterations,
    float keepProbability,
    float* input,
    float* result) {

    int indexInstance = blockIdx.x;

    int startInstanceWithinBatch = indexInstance * numberEntriesPerInstance;
    int startNextInstanceWithinBatch = startInstanceWithinBatch + numberEntriesPerInstance;

    int firstEntryWithinBatch = startInstanceWithinBatch + blockIdx.y * blockDim.x * numberIterations + threadIdx.x * numberIterations;

    if(firstEntryWithinBatch < startNextInstanceWithinBatch) {

        int lastEntryWithinBatch = min(firstEntryWithinBatch + numberIterations, startNextInstanceWithinBatch);

        if(indexInstance < batchSize) {

            for(int indexEntry = firstEntryWithinBatch; indexEntry < lastEntryWithinBatch; indexEntry++) {

                result[indexEntry] = keepProbability * input[indexEntry];

            }

        }
        else {

            setToZero(result, firstEntryWithinBatch, lastEntryWithinBatch);

        }

    }

}