#include "symbols/Nan.cuh"

extern "C"
__global__ void biasKernel (
    int batchSize,
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

    int firstEntryWithinBatch = startInstanceWithinBatch + startColumnWithinInstance + startRowWithinColumn;
    int startNextInstanceWithinBatch = startInstanceWithinBatch + numberEntriesPerInstance;

    if(firstEntryWithinBatch < startNextInstanceWithinBatch) {

        int lastEntryWithinBatch = min(firstEntryWithinBatch + numberIterations, startNextInstanceWithinBatch);

        if(indexInstance < batchSize) {

            for(int indexEntry = firstEntryWithinBatch; indexEntry < lastEntryWithinBatch; indexEntry++) {

                int indexColumn = indexEntry % numberRows;
                result[indexEntry] = input[indexEntry] + bias[indexColumn];

            }

        }
        else {

            setToNan(result, firstEntryWithinBatch, lastEntryWithinBatch);

        }

    }

}