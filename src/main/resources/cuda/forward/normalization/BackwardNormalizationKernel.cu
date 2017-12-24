#include "reduction/SumReduction.cuh"
#include "symbols/NaN.cuh"

__global__ void backwardNormalizationKernel (
    int batchSize,
    int numberRows,
    int numberEntriesPerInstance,
    int numberIterations,
    float* chain,
    float* forward,
    float* sums,
    float* result) {

    extern __shared__ float sharedData[];

    int indexInstance = blockIdx.x;
    int indexColumn = blockIdx.y;

    int startInstance = indexInstance * numberEntriesPerInstance;
    int startNextInstance = startInstance + numberEntriesPerInstance;

    int startColumnWithinInstance = indexColumn * numberRows;
    int startEntryWithinColumn = threadIdx.x * numberIterations;

    int firstEntryWithinBatch = startInstance + startColumnWithinInstance + startEntryWithinColumn;

    if(firstEntryWithinBatch < startNextInstance) {
        int lastEntryWithinBatch = min(firstEntryWithinBatch + numberIterations, startNextInstance);

        if(indexInstance < batchSize) {
            float thisValue = 0.0f;

            for(int index = firstEntryWithinBatch; index < lastEntryWithinBatch; index++) {
                thisValue += chain[index] * forward[index];
            }

            int warpId = threadIdx.x / warpSize;
            int laneId = threadIdx.x % warpSize;

            reduceToSum(thisValue, warpId, laneId, sharedData);

            int indexColumnInBatch = indexInstance * gridDim.y + indexColumn;

            for(int indexEntry = firstEntryWithinBatch; indexEntry < firstEntryWithinBatch + numberIterations; indexEntry++) {
                result[indexEntry] = (-sharedData[0] + chain[indexEntry]) / sums[indexColumnInBatch];
            }
        }
        else {
            setToNan(result, firstEntryWithinBatch, lastEntryWithinBatch);
        }
    }

}