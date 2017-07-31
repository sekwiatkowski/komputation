#include "reduction/Reduction.cuh"
#include "zero/Zero.cuh"

template <int blockSize>
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

    int startIndexWithinColumn = threadIdx.x * numberIterations;
    int startIndexWithinInstance = indexColumn * numberRows + startIndexWithinColumn;
    int startIndexWithinBatch = indexInstance * numberEntriesPerInstance + startIndexWithinInstance;

    if(indexInstance < batchSize) {

        int indexColumnInBatch = indexInstance * gridDim.y + indexColumn;

        if(startIndexWithinColumn < numberRows) {

            float sumOfProducts = 0.0;

            for(int indexEntry = startIndexWithinBatch; indexEntry < startIndexWithinBatch + numberIterations; indexEntry++) {

                sumOfProducts -= chain[indexEntry] * forward[indexEntry];

            }

            sharedData[threadIdx.x] = sumOfProducts;

        }
        else {

            sharedData[threadIdx.x] = 0.0;

        }

        __syncthreads();

        reduce<blockSize>(threadIdx.x, sharedData, 0);

        for(int indexEntry = startIndexWithinBatch; indexEntry < startIndexWithinBatch + numberIterations; indexEntry++) {

            result[indexEntry] = (sharedData[0] + chain[indexEntry]) / sums[indexColumnInBatch];

        }

    }
    else {

        setToZero(result, startIndexWithinBatch, numberIterations);

    }


}