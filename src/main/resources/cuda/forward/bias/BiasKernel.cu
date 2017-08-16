#include "zero/Zero.cuh"

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

    int startInstanceWithinBatch = indexInstance * numberEntriesPerInstance;
    int startNextInstanceWithinBatch = startInstanceWithinBatch + numberEntriesPerInstance;

    int firstEntryWithinBatch = startInstanceWithinBatch + blockIdx.y * blockDim.x * numberIterations + threadIdx.x * numberIterations;

    if(firstEntryWithinBatch < startNextInstanceWithinBatch) {

        int lastEntryWithinBatch = min(firstEntryWithinBatch + numberIterations, startNextInstanceWithinBatch);

        if(indexInstance < batchSize) {

            for(int indexEntry = firstEntryWithinBatch; indexEntry < lastEntryWithinBatch; indexEntry++) {

                int indexColumn = indexEntry % numberRows;
                result[indexEntry] = input[indexEntry] + bias[indexColumn];

            }

        }
        else {

            setToZero(result, firstEntryWithinBatch, lastEntryWithinBatch);

        }

    }

}