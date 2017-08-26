#include "symbols/Nan.cuh"

__device__ int findNextPowerOfTwo(int input) {

    return (int)powf(2.0, ceilf(log2f((float)input)));

}

__device__ int findMaximum(int thisIndex, float thisValue, int nextPowerOfTwo) {

    for (int offset = nextPowerOfTwo / 2; offset > 0; offset /= 2) {

        int otherIndex = __shfl_down(thisIndex, offset, nextPowerOfTwo);
        float otherValue = __shfl_down(thisValue, offset, nextPowerOfTwo);

        if(otherValue > thisValue) {

            thisIndex = otherIndex;
            thisValue = otherValue;

        }

    }

    return thisIndex;

}

__global__ void maxPoolingKernel (
    int batchSize,
    int* lengths,
    int numberEntries,
    int* maxIndices,
    float* input,
    float* result) {

    int indexInstance = blockIdx.x;
    int length = lengths[indexInstance];
    int indexRow = blockIdx.y;
    int indexColumn = threadIdx.x;
    int numberRows = gridDim.y;
    int maximumNumberOfColumns = blockDim.x;

    int resultStartInstance = indexInstance * numberRows;
    int resultIndex = resultStartInstance + indexRow;

    if(indexInstance < batchSize) {

        extern __shared__ int warpMaximumIndices[];

        int numberWarps = (maximumNumberOfColumns + warpSize - 1) / warpSize;
        int lastWarpId = numberWarps - 1;

        int warpId = indexColumn / warpSize;
        int laneId = indexColumn % warpSize;

        int startInstanceWithinBatch = indexInstance * numberEntries;
        int startColumnWithinBatch = indexColumn * numberRows;

        int thisIndex = startInstanceWithinBatch + startColumnWithinBatch + indexRow;

        float thisValue = warpId < lastWarpId ? input[thisIndex] : (indexColumn < length ? input[thisIndex] : __int_as_float(0xff800000));

        int width = warpId < lastWarpId ? warpSize : findNextPowerOfTwo(length);

        int warpMaximumIndex = findMaximum(thisIndex, thisValue, width);

        if(laneId == 0) {

            warpMaximumIndices[warpId] = warpMaximumIndex;

        }

        __syncthreads();

        if (warpId == 0 && laneId < numberWarps) {

            int thisWarpMaximumIndex = warpMaximumIndices[laneId];
            int thisWarpMaximumValue = input[thisWarpMaximumIndex];

            int blockMaximumIndex = findMaximum(thisWarpMaximumIndex, thisWarpMaximumValue, findNextPowerOfTwo(numberWarps));

            if(laneId == 0) {

                maxIndices[resultIndex] = blockMaximumIndex;
                result[resultIndex] = input[blockMaximumIndex];

            }

        }

    }
    else {

        maxIndices[resultIndex] = nanf("NaN");
        setToNan(result, resultStartInstance, resultStartInstance + 1);

    }

}