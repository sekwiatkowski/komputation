#include "reduction/SumReduction.cuh"

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

        int laneId = threadIdx.x % warpSize;

        // Go through each position within the filter
        if(laneId < filterLength) {

            int indexRowWithinInstance = indexEntryWithinInstance % numberRows;
            int indexColumnWithinInstance = indexEntryWithinInstance / numberRows;

            int indexRowWithinFilter = laneId % filterHeight;
            int indexColumnWithinFilter = laneId / filterHeight;

            // At which row does the convolution for the given filter position start
            int firstRowInConvolution = indexRowWithinInstance - indexRowWithinFilter;
            // At which row does the convolution for the given filter position end
            int lastRowInConvolution = firstRowInConvolution + filterHeight - 1;

            // At which column does the convolution for the given filter position start
            int firstColumnInConvolution = indexColumnWithinInstance - indexColumnWithinFilter;

            // At which column does the convolution for the given filter position end
            int lastColumnInConvolution = firstColumnInConvolution + filterWidth - 1;

            float thisValue;

            if(firstRowInConvolution >= 0 && lastRowInConvolution < numberRows &&
               firstColumnInConvolution >= 0 && lastColumnInConvolution < lengths[indexInstance]) {

                int indexConvolution = firstColumnInConvolution * convolutionsPerRow + firstRowInConvolution;

                int indexGradient = indexInstance * maximumConvolutions + indexConvolution * filterLength + indexColumnWithinFilter * filterHeight + indexRowWithinFilter;

                thisValue = gradient[indexGradient];

            }
            else {

                thisValue = 0.0;

            }

            float sum = warpReduceToSum(thisValue);

            if(laneId == 0) {

                result[indexEntryWithinBatch] = sum;

            }

        }

    }
    else {

        result[indexEntryWithinBatch] = nan("NaN");

    }

}
