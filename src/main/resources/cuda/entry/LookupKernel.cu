#include "symbols/NaN.cuh"

__global__ void lookupKernel (
    float* vectors,
    int* vectorIds,
    float* result,
    int maximumBatchSize,
    int maximumLength,
    int dimension,
    int numberIterations) {

    int startEntryWithinColumn = threadIdx.x * numberIterations;

    if(startEntryWithinColumn < dimension) {
        int indexInstance = blockIdx.x;
        int indexColumn = blockIdx.y;

        int startInstanceWithinBatch = indexInstance * maximumLength * dimension;
        int startColumnWithinInstance = indexColumn * dimension;
        int startColumnWithinBatch = startInstanceWithinBatch + startColumnWithinInstance;

        int firstEntryWithinBatch = startColumnWithinBatch + startEntryWithinColumn;
        int lastEntryWithinBatch = min(firstEntryWithinBatch + numberIterations, startColumnWithinBatch + dimension);

        int indexVectorId = indexInstance * maximumLength + indexColumn;
        int vectorId = vectorIds[indexVectorId];

        if(indexInstance < maximumBatchSize && vectorId != -1) {
            int startWithinLookupTable = vectorId * dimension + startEntryWithinColumn;

            for(int indexResult = firstEntryWithinBatch, indexLookupTable = startWithinLookupTable; indexResult < lastEntryWithinBatch; indexResult++, indexLookupTable++) {
                result[indexResult] = vectors[indexLookupTable];
            }
        }
        else {
            setToNan(result, firstEntryWithinBatch, lastEntryWithinBatch);
        }
    }

}