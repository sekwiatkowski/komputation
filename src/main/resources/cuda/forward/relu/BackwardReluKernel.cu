#include "symbols/Zero.cuh"

__device__ float backwardRelu (float forward, float chain)
{

    if(forward > 0.0) {

        return chain;

    }
    else {

        return 0.0;

    }

}

extern "C"
__global__ void backwardReluKernel (int batchSize, int numberEntriesPerInstance, int numberIterations, float *forward, float *chain, float *destination) {

    int indexInstance = blockIdx.x;

    int startInstanceWithinBatch = indexInstance * numberEntriesPerInstance;
    int startNextInstanceWithinBatch = startInstanceWithinBatch + numberEntriesPerInstance;

    int firstEntryWithinBatch = startInstanceWithinBatch + blockIdx.y * blockDim.x * numberIterations + threadIdx.x * numberIterations;

    if(firstEntryWithinBatch < startNextInstanceWithinBatch) {

        int lastEntryWithinBatch = min(firstEntryWithinBatch + numberIterations, startNextInstanceWithinBatch);

        if(indexInstance < batchSize) {

            for(int indexEntry = firstEntryWithinBatch; indexEntry < lastEntryWithinBatch; indexEntry++) {

                destination[indexEntry] = backwardRelu(forward[indexEntry], chain[indexEntry]);

            }

        }
        else {

            setToZero(destination, firstEntryWithinBatch, lastEntryWithinBatch);

        }

    }

}