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
    int* inputLengths,
    int numberRows,
    int numberFilterRowPositions,
    int numberInputEntries,
    int numberResultEntries,
    int filterHeight,
    int filterWidth,
    int filterLength,
    float* input,
    float* result,
    int* resultLengths) {

    int indexInstance = blockIdx.x;

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

        int inputLength = inputLengths[indexInstance];

        int numberConvolutions = (inputLength - filterWidth + 1) * numberFilterRowPositions;

        if(indexConvolution < numberConvolutions) {

            result[indexEntryWithinResult] = input[indexInstance * numberInputEntries + indexColumn * numberRows + indexRow];

        }
        else {

            result[indexEntryWithinResult] = 0.0;

        }

        if(blockIdx.y == 0 && threadIdx.x == 0) {

            resultLengths[indexInstance] = numberConvolutions;

        }

    }
    else {

        result[indexEntryWithinResult] = nanf("NaN");

        resultLengths[indexInstance] = 0;

    }

}