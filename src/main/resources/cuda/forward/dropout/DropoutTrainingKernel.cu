#include "symbols/NaN.cuh"

__device__  int xorShift(int seed) {

    int updated = seed;

    updated ^= updated << 13;
    updated ^= updated >> 17;
    updated ^= updated << 5;

    return updated;

}

/*
    dropout probability is 1 - keep probability and should be less than 1.
    Adding 2147483648 ensures that the result is non-negative (from 0 to 2 * upper integer bound).
    Division by 4294967295.0 returns a percentage (from 0 to 1).
    Subtraction by the probability dropout probability returns either a positive or a negative number.
    Drop out if the number is negative.
*/
__device__ float generateMask(float seed, float dropoutProbability) {

    return ceilf((seed + 2147483648.0) / 4294967295.0 - dropoutProbability);

}

__global__ void dropoutTrainingKernel (
    int batchSize,
    int numberRows,
    int numberEntriesPerInstance,
    int numberIterations,
    float dropoutProbability,
    float* input,
    int* seeds,
    float* masks,
    float* result) {

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

                int newSeed = xorShift(seeds[indexEntry]);
                seeds[indexEntry] = newSeed;

                float mask = generateMask((float)newSeed, dropoutProbability);
                masks[indexEntry] = mask;

                result[indexEntry] = mask * input[indexEntry];

            }

        }
        else {

            setToNan(result, firstEntryWithinBatch, lastEntryWithinBatch);

        }

    }

}