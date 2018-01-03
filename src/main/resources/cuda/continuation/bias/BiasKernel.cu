#include "symbols/NaN.cuh"

__global__ void biasKernel (
    int batchSize,
    int* lengths,
    int numberEntriesPerInstance,
    int numberRows,
    int numberIterations,
    float* input,
    float* bias,
    float* result) {
    int indexInstance = blockIdx.x;
    int indexColumn = blockIdx.y;

    int startInstanceWithinBatch = indexInstance * numberEntriesPerInstance;
    int startColumnWithinInstance = indexColumn * numberRows;
    int startRowWithinColumn = threadIdx.x * numberIterations;

    int firstEntryWithinInstance = startColumnWithinInstance + startRowWithinColumn;

    if(firstEntryWithinInstance < numberEntriesPerInstance) {
        int startColumnWithinBatch = startInstanceWithinBatch + startColumnWithinInstance;
        int firstEntryWithinBatch = startColumnWithinBatch + startRowWithinColumn;

        int startNextColumnWithinBatch = startColumnWithinBatch + numberRows;
        int lastEntryWithinBatch = min(firstEntryWithinBatch + numberIterations, startNextColumnWithinBatch);

        if(indexInstance < batchSize) {
            int length = lengths[indexInstance];

            if(indexColumn < length) {
                for(int indexEntry = firstEntryWithinBatch; indexEntry < lastEntryWithinBatch; indexEntry++) {
                    int indexColumn = indexEntry % numberRows;
                    result[indexEntry] = input[indexEntry] + bias[indexColumn];
                }
            }
            else {
                setToNaN(result, firstEntryWithinBatch, lastEntryWithinBatch);
            }
        }
        else {
            setToNaN(result, firstEntryWithinBatch, lastEntryWithinBatch);
        }
    }
}