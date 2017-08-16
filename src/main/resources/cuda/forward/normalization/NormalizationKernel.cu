#include "reduction/SumReduction.cuh"
#include "zero/Zero.cuh"

/*
    number of blocks in x-dimension = number of instances
    number of blocks in y-dimension = number of columns
    number threads per block = number of rows
*/

__global__ void normalizationKernel (
    int batchSize,
    int numberRows,
    int numberEntriesPerInstance,
    int numberIterations,
    float* input,
    float* sums,
    float* result) {

    extern __shared__ float sharedData[];

    int indexInstance = blockIdx.x;
    int indexColumn = blockIdx.y;
    int numberColumns = gridDim.y;
    int threadId = threadIdx.x;

    int startInstance = indexInstance * numberEntriesPerInstance;
    int startNextInstance = startInstance + numberEntriesPerInstance;

    int startColumnWithinInstance = indexColumn * numberRows;
    int startEntryWithinColumn = threadIdx.x * numberIterations;

    int firstEntryWithinBatch = startInstance + startColumnWithinInstance + startEntryWithinColumn;

    if(firstEntryWithinBatch < startNextInstance) {

        int indexColumnInBatch = indexInstance * numberColumns + indexColumn;
        int lastEntryWithinBatch = min(firstEntryWithinBatch + numberIterations, startNextInstance);

        if(indexInstance < batchSize) {

            float thisValue = input[firstEntryWithinBatch];

            if(numberIterations > 1) {

                for(int index = firstEntryWithinBatch + 1; index < lastEntryWithinBatch; index++) {

                    thisValue += input[index];

                }

            }

            int warpId = threadId / warpSize;
            int laneId = threadId % warpSize;

            reduceToSum(thisValue, warpId, laneId, sharedData);

            result[firstEntryWithinBatch] = input[firstEntryWithinBatch] / sharedData[0];

            if(numberIterations > 1) {

                for(int indexEntry = firstEntryWithinBatch+1; indexEntry < lastEntryWithinBatch; indexEntry++) {

                    result[indexEntry] = input[indexEntry] / sharedData[0];

                }

            }

            if(threadId == 0) {

                sums[indexColumnInBatch] = sharedData[0];

            }

        }
        else {

            setToZero(result, firstEntryWithinBatch, lastEntryWithinBatch);

            if(threadId == 0) {

                sums[indexColumnInBatch] = 0.0;

            }

        }

    }

}