#include "zero/Zero.cuh"

__device__  int xorShift(int seed) {

    int updated = seed;

    updated ^= updated << 13;
    updated ^= updated >> 17;
    updated ^= updated << 5;

    return updated;

}

/*
    dropout probability is 1 - keep probability and should be less than 1.
    seed + 2147483648.0: [0, 2^32/2 + 2^32/2-1 = 4294967295]
    (seed + 2147483648.0) / 4294967295.0: [0 to 1]
    (seed + 2147483648.0) / 4294967295.0 - dropout probability): (0 to 1]
    ceilf(seed + 2147483648.0) / 4294967295.0 - dropout probability): or or 1
*/

__device__ float generateMask(float seed, float dropoutProbability) {

    return ceilf((seed + 2147483648.0) / 4294967295.0 - dropoutProbability);

}

extern "C"
__global__ void dropoutTrainingKernel (int batchSize, int numberEntriesPerInstance, int numberIterations, float dropoutProbability, float* input, int* seeds, float* masks, float* result)
{

    // What's the first entry index within the instance that this thread should operate on?
    int startIndexWithinInstance = blockIdx.y * (blockDim.x * numberIterations) + threadIdx.x * numberIterations;

    // Continue if this index is smaller than the dimension of the instance.
    if(startIndexWithinInstance < numberEntriesPerInstance) {

        // What's the first entry index within the batch that this thread should operate on?
        int startIndexWithinBatch = blockIdx.x * numberEntriesPerInstance + startIndexWithinInstance;

        // Is the instance greater than the current batch size?
        if(blockIdx.x >= batchSize) {

            setToZero(result, startIndexWithinBatch, numberIterations);

        }
        else {

            for(int indexEntry = startIndexWithinBatch; indexEntry < startIndexWithinBatch + numberIterations; indexEntry++) {

                int newSeed = xorShift(seeds[indexEntry]);
                seeds[indexEntry] = newSeed;

                float mask = generateMask((float)newSeed, dropoutProbability);
                masks[indexEntry] = mask;

                result[indexEntry] = mask * input[indexEntry];

            }

        }

    }

}