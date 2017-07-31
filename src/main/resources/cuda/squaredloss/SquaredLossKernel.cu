#include "reduction/Reduction.cuh"

// This assumes that the number of threads is equal to the number of predictions/targets.
// First, the squared differences between predictions and targets are stored in shared memory.
// In the second step, the squared differences are summed up using a parallel reduction.
// Finally, the sum is multiplied by 1/2.
template <int blockSize>
__global__ void squaredLossKernel (int batchSize, int numberRows, int numberEntriesPerInstance, int numberIterations, float* predictions, float* targets, float* result)
{

    int startIndexWithinColumn = threadIdx.x * numberIterations;

    extern __shared__ float sharedData[];

    int indexInstance = blockIdx.x;
    int indexColumn = blockIdx.y;

    int startIndexWithinInstance = indexColumn * numberRows + startIndexWithinColumn;
    int startIndexWithinBatch = indexInstance * numberEntriesPerInstance + startIndexWithinInstance;

    int indexColumnInBatch = indexInstance * gridDim.y + indexColumn;

    if(indexInstance < batchSize) {

        if(startIndexWithinColumn < numberRows) {

            float sum = 0.0f;

            for(int indexEntry = startIndexWithinBatch; indexEntry < startIndexWithinBatch + numberIterations; indexEntry++) {

                sum  += powf(predictions[indexEntry] - targets[indexEntry], 2.0);

            }

            sharedData[threadIdx.x] = sum;

        }
        else {

            sharedData[threadIdx.x] = 0.0;

        }

        __syncthreads();

        reduce<blockSize>(threadIdx.x, sharedData, 0);

        if(threadIdx.x == 0) {

            result[indexColumnInBatch] = sharedData[0];

        }

    }
    else {

        if(threadIdx.x == 0) {

            result[indexColumnInBatch] = 0.0;

        }

    }


}