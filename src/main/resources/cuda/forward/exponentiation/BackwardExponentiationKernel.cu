#include "zero/Zero.cuh"

extern "C"
__global__ void backwardExponentiationKernel (
    int batchSize,
    int numberEntriesPerInstance,
    int numberIterations,
    float *forward,
    float *chain,
    float *destination) {

    int indexInstance = blockIdx.x;

    int startInstanceWithinBatch = indexInstance * numberEntriesPerInstance;
    int startNextInstanceWithinBatch = startInstanceWithinBatch + numberEntriesPerInstance;

    int firstEntryWithinBatch = startInstanceWithinBatch + blockIdx.y * blockDim.x * numberIterations + threadIdx.x * numberIterations;

    if(firstEntryWithinBatch < startNextInstanceWithinBatch) {

        int lastEntryWithinBatch = min(firstEntryWithinBatch + numberIterations, startNextInstanceWithinBatch);

        if(indexInstance < batchSize) {

            for(int indexEntry = firstEntryWithinBatch; indexEntry < lastEntryWithinBatch; indexEntry++) {

                destination[indexEntry] = chain[indexEntry] * forward[indexEntry];

            }

        }
        else {

            setToZero(destination, firstEntryWithinBatch, lastEntryWithinBatch);

        }

    }

}