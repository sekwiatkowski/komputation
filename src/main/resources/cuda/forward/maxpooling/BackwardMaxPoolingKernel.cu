#include "zero/Zero.cuh"

__global__ void backwardMaxPoolingKernel (
    int batchSize,
    int numberEntriesPerInstance,
    int numberRows,
    int* maxIndices,
    float* chain,
    float* result) {

    int indexInstance = blockIdx.x;
    int startInstance = indexInstance * numberEntriesPerInstance;

    int indexColumn = threadIdx.x / numberRows;
    int startColumnWithinInstance = indexColumn * numberRows;

    int indexEntryWithinBatch = startInstance + startColumnWithinInstance + threadIdx.x;

    result[indexEntryWithinBatch] = 0.0;

    if(threadIdx.x == 0) {

        int indexRow = blockIdx.y;

        int maxIndex = maxIndices[indexRow];

        result[maxIndex] = chain[indexRow];

    }

}