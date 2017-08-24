#include "symbols/Nan.cuh"

extern "C"
__global__ void backwardDropoutKernel (
    int batchSize,
    int numberEntriesPerInstance,
    int numberRows,
    int numberIterations,
    float* chain,
    float* mask,
    float* result) {

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

                result[indexEntry] = chain[indexEntry] * mask[indexEntry];

            }

        }
        else {

            setToNan(result, firstEntryWithinBatch, lastEntryWithinBatch);

        }

    }

}