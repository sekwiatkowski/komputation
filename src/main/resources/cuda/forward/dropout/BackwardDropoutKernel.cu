#include "symbols/Zero.cuh"

extern "C"
__global__ void backwardDropoutKernel (int batchSize, int numberEntriesPerInstance, int numberIterations, float* chain, float* mask, float* result)
{

    int indexInstance = blockIdx.x;

    int startInstanceWithinBatch = indexInstance * numberEntriesPerInstance;
    int startNextInstanceWithinBatch = startInstanceWithinBatch + numberEntriesPerInstance;

    int firstEntryWithinBatch = startInstanceWithinBatch + blockIdx.y * blockDim.x * numberIterations + threadIdx.x * numberIterations;

    if(firstEntryWithinBatch < startNextInstanceWithinBatch) {

        int lastEntryWithinBatch = min(firstEntryWithinBatch + numberIterations, startNextInstanceWithinBatch);

        if(indexInstance < batchSize) {

            for(int indexEntry = firstEntryWithinBatch; indexEntry < lastEntryWithinBatch; indexEntry++) {

                result[indexEntry] = chain[indexEntry] * mask[indexEntry];

            }

        }
        else {

            setToZero(result, firstEntryWithinBatch, lastEntryWithinBatch);

        }

    }

}