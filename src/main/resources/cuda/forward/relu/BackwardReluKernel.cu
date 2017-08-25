#include "symbols/Nan.cuh"

__device__ float backwardRelu (float forward, float chain)
{

    if(forward > 0.0) {

        return chain;

    }
    else {

        return 0.0;

    }

}

__global__ void backwardReluKernel (
    int batchSize,
    int numberRows,
    int numberEntriesPerInstance,
    int numberIterations,
    float *forward,
    float *chain,
    float *destination) {

    int indexInstance = blockIdx.x;
    int indexColumn = blockIdx.y;

    int startInstanceWithinBatch = indexInstance * numberEntriesPerInstance;
    int startColumnWithinInstance = indexColumn * numberRows;
    int startRowWithinColumn = threadIdx.x * numberIterations;

    int firstEntryWithinBatch = startInstanceWithinBatch + startColumnWithinInstance + startRowWithinColumn;
    int startNextColumn = startInstanceWithinBatch + startColumnWithinInstance + numberRows;

    if(firstEntryWithinBatch < startNextColumn) {

        int lastEntryWithinBatch = min(firstEntryWithinBatch + numberIterations, startNextColumn);

        if(indexInstance < batchSize) {

            for(int indexEntry = firstEntryWithinBatch; indexEntry < lastEntryWithinBatch; indexEntry++) {

                destination[indexEntry] = backwardRelu(forward[indexEntry], chain[indexEntry]);

            }

        }
        else {

            setToNan(destination, firstEntryWithinBatch, lastEntryWithinBatch);

        }

    }

}