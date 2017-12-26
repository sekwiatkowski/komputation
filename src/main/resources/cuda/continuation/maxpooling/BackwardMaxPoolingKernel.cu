__global__ void backwardMaxPoolingKernel (
    int batchSize,
    int* lengths,
    float symbolForUnusedColumns,
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

    if(indexInstance < batchSize) {
        int length = lengths[indexInstance];

        if(indexColumn < length) {

            int maxIndex = maxIndices[indexInstance * numberRows + indexRow];

            if(indexEntryWithinBatch == maxIndex) {
                result[indexEntryWithinBatch] = chain[indexInstance * numberRows + indexRow];
            }
            else {
                result[indexEntryWithinBatch] = 0.0;
            }

        }
        else {
            result[indexEntryWithinBatch] = symbolForUnusedColumns;
        }
    }
    else {
        result[indexEntryWithinBatch] = symbolForUnusedColumns;
    }

}