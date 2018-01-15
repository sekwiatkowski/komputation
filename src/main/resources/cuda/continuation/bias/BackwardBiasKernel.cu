#include "../../cuda.h"
#include "../../reduction/SumReduction.cuh"

/*
    Number of blocks = number of chain rows
    Number of threads = (number of chain columns + number of iterations - 1) / number of iterations
    Sum-reduce each row, use the first thread in a block to set the result entry

    1 1
    2 2
    3 3
    4 4
    5 5
    6 6

    number of blocks = 6
*/

__global__ void backwardBiasKernel (
    float* chain,
    int numberChainRows,
    int numberChainColumns,
    int numberIterations,
    float* result) {

    extern __shared__ float sharedData[];

    int chainRowIndex = blockIdx.x;
    int firstChainColumnIndex = threadIdx.x * numberIterations;
    int lastChainColumnIndex = min(firstChainColumnIndex + numberIterations, numberChainColumns);

    float thisValue = 0.0;
    int chainColumnIndex = firstChainColumnIndex;

    while(chainColumnIndex < lastChainColumnIndex) {
        float chainEntry = chain[chainColumnIndex * numberChainRows + chainRowIndex];

        float toBeAdded = isnan(chainEntry) ? 0 : chainEntry;

        thisValue += toBeAdded;

        chainColumnIndex++;
    }

    __syncthreads();

    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    float sum = reduceWarpsToSum(thisValue, warpId, laneId, sharedData);

    if(threadIdx.x == 0) {
        result[blockIdx.x] = sum;
    }

    __syncthreads();

}