#include "reduction/SumReduction.cuh"

__global__ void backwardExpansionKernel(
    int batchSize,
    int* lengths,
    int numberIterations,
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
    int indexInstance = blockIdx.x;
    int startInstanceWithinBatch = indexInstance * numberEntries;
    int firstEntryInYBlockWithinInstance = blockIdx.y * numberOfWarpsPerBlocks;
    int indexEntryWithinYBlock = threadIdx.x / warpSize;

    int indexEntryWithinInstance = firstEntryInYBlockWithinInstance + indexEntryWithinYBlock;

    if(indexEntryWithinInstance < numberEntries) {
        int length = lengths[indexInstance];

        int indexEntryWithinBatch = startInstanceWithinBatch + indexEntryWithinInstance;
        result[indexEntryWithinBatch] = nanf("NaN");

        if(indexInstance < batchSize && indexEntryWithinInstance < length * numberRows) {
            int laneId = threadIdx.x % warpSize;

            int startFilter = laneId * numberIterations;
            int endFilter = min(startFilter + numberIterations, filterLength);

            int indexRowWithinInstance = indexEntryWithinInstance % numberRows;
            int indexColumnWithinInstance = indexEntryWithinInstance / numberRows;

            float thisValue = 0.0;

            for(int indexFilter = startFilter; indexFilter < endFilter; indexFilter++) {
                int indexRowWithinFilter = indexFilter % filterHeight;
                int indexColumnWithinFilter = indexFilter / filterHeight;

                // At which row does the convolution for the given filter position start
                int firstRowInConvolution = indexRowWithinInstance - indexRowWithinFilter;
                // At which row does the convolution for the given filter position end
                int lastRowInConvolution = firstRowInConvolution + filterHeight - 1;

                // At which column does the convolution for the given filter position start
                int firstColumnInConvolution = indexColumnWithinInstance - indexColumnWithinFilter;
                // At which column does the convolution for the given filter position end
                int lastColumnInConvolution = firstColumnInConvolution + filterWidth - 1;

                float thisValueInIteration;

                if(firstRowInConvolution >= 0 && lastRowInConvolution < numberRows &&
                   firstColumnInConvolution >= 0 && lastColumnInConvolution < lengths[indexInstance]) {

                    int indexConvolution = firstColumnInConvolution * convolutionsPerRow + firstRowInConvolution;

                    int indexGradient = indexInstance * maximumConvolutions * filterLength + indexConvolution * filterLength + indexColumnWithinFilter * filterHeight + indexRowWithinFilter;

                    thisValueInIteration = gradient[indexGradient];

                }
                else {
                    thisValueInIteration = 0.0;
                }

                thisValue += thisValueInIteration;
            }

            float sum = warpReduceToSum(thisValue);

            if(laneId == 0) {
                result[indexEntryWithinBatch] = sum;
            }
        }
    }
}
