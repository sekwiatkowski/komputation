__global__ void backwardMaxPoolingKernel (
    int batchSize,
    int numberEntriesPerInstance,
    int numberRows,
    int* maxIndices,
    float* chain,
    float* result) {

    int indexInstance = blockIdx.x;
    int indexRow = blockIdx.y;
    int indexColumn = threadIdx.x;

    int startInstanceWithinBatch = indexInstance * numberEntriesPerInstance;
    int startColumnWithinInstance = indexColumn * numberRows;
    int indexEntryWithinBatch = startInstanceWithinBatch + startColumnWithinInstance + indexRow;

    result[indexEntryWithinBatch] = 0.0;

    int maxIndexWithinRow = maxIndices[indexRow];

    if(indexEntryWithinBatch == maxIndexWithinRow) {

        result[indexEntryWithinBatch] = chain[indexRow];

    }

}