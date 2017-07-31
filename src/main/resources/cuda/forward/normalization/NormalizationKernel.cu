#include "reduction/Reduction.cuh"
#include "zero/Zero.cuh"

template <int blockSize>
__global__ void normalizationKernel (int batchSize, int numberRows, int numberEntriesPerInstance, int numberIterations, float* input, float* sums, float* result)
{

    int startIndexWithinColumn = threadIdx.x * numberIterations;

    extern __shared__ float sharedData[];

    int indexColumn = blockIdx.y;
    int startIndexWithinInstance = indexColumn * numberRows + startIndexWithinColumn;

    int indexInstance = blockIdx.x;
    int startIndexWithinBatch = indexInstance * numberEntriesPerInstance + startIndexWithinInstance;

    int indexColumnInBatch = indexInstance * gridDim.y + indexColumn;

    if(indexInstance < batchSize) {

        if(startIndexWithinColumn < numberRows) {

            float sum = 0.0f;

            for(int indexEntry = startIndexWithinBatch; indexEntry < startIndexWithinBatch + numberIterations; indexEntry++) {

                sum  += input[indexEntry];

            }

            sharedData[threadIdx.x] = sum;

        }
        else {

            sharedData[threadIdx.x] = 0.0;

        }

        __syncthreads();

        reduce<blockSize>(threadIdx.x, sharedData, 0);

        for(int indexEntry = startIndexWithinBatch; indexEntry < startIndexWithinBatch + numberIterations; indexEntry++) {

            result[indexEntry] = input[indexEntry] / sharedData[0];

        }

        if(threadIdx.x == 0) {

            sums[indexColumnInBatch] = sharedData[0];

        }

    }
    else {

        setToZero(result, startIndexWithinBatch, numberIterations);

        if(threadIdx.x == 0) {

            sums[indexColumnInBatch] = 0.0;

        }

    }


}