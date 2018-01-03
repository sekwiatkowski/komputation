#include "symbols/NaN.cuh"

__inline__ __device__ int findNextPowerOfTwo(int input) {

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
    extern __shared__ int warpMaximumIndices[];

    // One instance per block in the X dimension
    int indexInstance = blockIdx.x;
    // One row per block in the Y dimension
    int indexRow = blockIdx.y;
    // One column per thread
    int indexColumn = threadIdx.x;

    int numberRows = gridDim.y;

    int resultStartInstance = indexInstance * numberRows;
    int resultIndex = resultStartInstance + indexRow;

    if(indexInstance < batchSize) {
        int length = lengths[indexInstance];

        int warpId = indexColumn / warpSize;

        int numberRequiredWarps = (length + warpSize - 1) / warpSize;

        // Some instances/rows require more warps than others
        if(warpId < numberRequiredWarps) {
            int laneId = indexColumn % warpSize;

            int thisIndex = indexInstance * numberEntries + indexColumn * numberRows + indexRow;
            float thisValue = indexColumn < length ? input[thisIndex] : __int_as_float(0xff800000);

            int lastWarpId = numberRequiredWarps - 1;
            int width = warpId < lastWarpId ? warpSize : findNextPowerOfTwo(length - lastWarpId * warpSize);

            int warpMaximumIndex = findMaximum(thisIndex, thisValue, width);

            if(laneId == 0) {
                warpMaximumIndices[warpId] = warpMaximumIndex;
            }

            __syncthreads();

            if (warpId == 0 && laneId < numberRequiredWarps) {
                int warpMaximumIndex = warpMaximumIndices[laneId];
                float warpMaximumValue = input[warpMaximumIndex];

                int blockMaximumIndex = findMaximum(warpMaximumIndex, warpMaximumValue, findNextPowerOfTwo(numberRequiredWarps));

                if(laneId == 0) {
                    maxIndices[resultIndex] = blockMaximumIndex;
                    result[resultIndex] = input[blockMaximumIndex];
                }
            }
        }
    }
    else {
        maxIndices[resultIndex] = nanf("NaN");
        setToNaN(result, resultStartInstance, resultStartInstance + 1);
    }

}