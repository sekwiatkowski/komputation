#include "../../cuda.h"
#include "../../symbols/NaN.cuh"
#include "Sigmoid.cuh"

__global__ void backwardSigmoidKernel (
    int batchSize,
    int numberRows,
    int numberEntriesPerInstance,
    int numberIterations,
    float* forward,
    float* chain,
    float* destination) {

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
                destination[indexEntry] = backwardSigmoid(forward[indexEntry], chain[indexEntry]);
            }
        }
        else {
            setToNaN(destination, firstEntryWithinBatch, lastEntryWithinBatch);
        }
    }
}