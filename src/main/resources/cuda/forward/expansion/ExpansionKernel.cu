#include "symbols/Nan.cuh"
#include "symbols/Zero.cuh"

/*
    1 4 7
    2 5 8
    3 6 9

    2*2 filter

    Number of blocks in y dimensions: number of columns positions
    Number of threads per block: number of row positions

*/
__global__ void expansionKernel(
    int batchSize,
    int* batchLengths,
    int numberRows,
    int numberFilterRowPositions,
    int numberInputEntries,
    int numberResultEntries,
    int filterHeight,
    int filterWidth,
    int filterLength,
    float* input,
    float* result) {

    int indexInstance = blockIdx.x;
    int length = batchLengths[indexInstance];

    int indexConvolution = blockIdx.y;
    int indexConvolutionEntry = threadIdx.x;

    int firstColumnOfConvolution = indexConvolution / numberFilterRowPositions;
    int firstRowOfConvolution = indexConvolution % numberFilterRowPositions;

    int relativeIndexColumn = indexConvolutionEntry / filterHeight;
    int relativeIndexRow = indexConvolutionEntry % filterHeight;

    int indexColumn = firstColumnOfConvolution + relativeIndexColumn;
    int indexRow = firstRowOfConvolution + relativeIndexRow;

    int indexEntryWithinResult = indexInstance * numberResultEntries + indexConvolution * filterLength + indexConvolutionEntry;

    if(indexInstance < batchSize) {

        int numberConvolutions = (length - filterWidth + 1) * numberFilterRowPositions;

        if(indexConvolution < numberConvolutions) {

            result[indexEntryWithinResult] = input[indexInstance * numberInputEntries + indexColumn * numberRows + indexRow];

        }
        else {

            result[indexEntryWithinResult] = 0.0;

        }

    }
    else {

        result[indexEntryWithinResult] = nanf("NaN");

    }

}
