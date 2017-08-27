#include "reduction/SumReduction.cuh"

/*

    1 4 7     1 4   2 5   4 7   5 8
    2 5 8     2 5   3 6   5 8   6 9
    3 6 9

*/

__global__ void backwardExpansionKernel(
    int batchSize,
    int* lengths,
    int numberRows,
    int numberEntries,
    int numberOfWarpsPerBlocks,
    int filterHeight,
    int filterWidth,
    int filterLength,
    int maximumConvolutions,
    int convolutionsPerRow,
    float* gradient,
    float* result) {

    extern __shared__ float sharedData[];

    int indexInstance = blockIdx.x;
    int startInstanceWithinBatch = indexInstance * numberEntries;
    int firstEntryInYBlockWithinInstance = blockIdx.y * numberOfWarpsPerBlocks;
    int indexEntryWithinYBlock = threadIdx.x / warpSize;

    int indexEntryWithinInstance = firstEntryInYBlockWithinInstance + indexEntryWithinYBlock;
    int indexEntryWithinBatch = startInstanceWithinBatch + indexEntryWithinInstance;

    if(indexInstance < batchSize) {

        int indexRow = indexEntryWithinInstance % numberRows;
        int indexColumn = indexEntryWithinInstance / numberRows;

        int warpId = threadIdx.x / warpSize;
        int laneId = threadIdx.x % warpSize;

        if(laneId < filterLength) {

            int indexRowWithinFilter = laneId % filterHeight;
            int indexColumnWithinFilter = laneId / filterHeight;

            int firstRowInConvolution = indexRow - indexRowWithinFilter;
            int lastRowInConvolution = firstRowInConvolution + filterHeight - 1;

            int firstColumnInConvolution = indexColumn - indexColumnWithinFilter;
            int lastColumnInConvolution = firstColumnInConvolution + filterWidth - 1;

            float thisValue;

            if(firstRowInConvolution >= 0 && lastRowInConvolution < numberRows && firstColumnInConvolution >= 0 && lastColumnInConvolution < lengths[indexInstance]) {

                int indexConvolution = firstColumnInConvolution * convolutionsPerRow + firstRowInConvolution;

                int indexGradient = indexInstance * maximumConvolutions + indexConvolution * filterLength + indexColumnWithinFilter * filterHeight + indexRowWithinFilter;

                thisValue = gradient[indexGradient];

            }
            else {

                thisValue = 0.0;

            }

            sharedData[laneId] = thisValue;

            __syncthreads();

            reduceToSum(thisValue, warpId, laneId, sharedData);

            if(laneId == 0) {

                result[indexEntryWithinBatch] = sharedData[0];

            }

        }

    }
    else {

        result[indexEntryWithinBatch] = nan("NaN");

    }

}
