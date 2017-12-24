#include "symbols/NaN.cuh"

__global__ void exponentiationKernel (
    int batchSize,
    int numberRows,
    int numberEntriesPerInstance,
    int numberIterations,
    float *source,
    float *destination) {
    int indexInstance = blockIdx.x;
    int indexColumn = blockIdx.y;

    int startInstanceWithinBatch = indexInstance * numberEntriesPerInstance;
    int startColumnWithinInstance = indexColumn * numberRows;
    int startRowWithinColumn = threadIdx.x * numberIterations;

    int startColumnWithinBatch = startInstanceWithinBatch + startColumnWithinInstance;

    int firstEntryWithinBatch = startColumnWithinBatch + startRowWithinColumn;
    int startNextColumn = startColumnWithinBatch + numberRows;

    if(firstEntryWithinBatch < startNextColumn) {
        int lastEntryWithinBatch = min(firstEntryWithinBatch + numberIterations, startNextColumn);

        if(indexInstance < batchSize) {
            for(int indexEntry = firstEntryWithinBatch; indexEntry < lastEntryWithinBatch; indexEntry++) {
                destination[indexEntry] = expf(source[indexEntry]);
            }
        }
        else {
            setToNan(destination, firstEntryWithinBatch, lastEntryWithinBatch);
        }
    }
}