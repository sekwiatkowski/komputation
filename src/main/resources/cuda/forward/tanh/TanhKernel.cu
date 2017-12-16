#include "symbols/NaN.cuh"

__device__ float tanh (float x) {
    return (2.0 / (1.0 + expf(-2.0*x))) - 1.0;
}

__global__ void tanhKernel (
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

    int firstEntryWithinBatch = startInstanceWithinBatch + startColumnWithinInstance + startRowWithinColumn;
    int startNextColumn = startInstanceWithinBatch + startColumnWithinInstance + numberRows;

    if(firstEntryWithinBatch < startNextColumn) {
        int lastEntryWithinBatch = min(firstEntryWithinBatch + numberIterations, startNextColumn);

        if(indexInstance < batchSize) {
            for(int indexEntry = firstEntryWithinBatch; indexEntry < lastEntryWithinBatch; indexEntry++) {
                destination[indexEntry] = tanh(source[indexEntry]);
            }
        }
        else {
            setToNan(destination, firstEntryWithinBatch, lastEntryWithinBatch);
        }
    }

}