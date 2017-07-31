#include "zero/Zero.cuh"

__device__ float relu (float x)
{

    return fmaxf(x, 0.0);

}

extern "C"
__global__ void reluKernel (int batchSize, int numberEntriesPerInstance, int numberIterations, float *source, float *destination) {

    // What's the first entry index within the instance that this thread should operate on?
    int startIndexWithinInstance = blockIdx.y * (blockDim.x * numberIterations) + threadIdx.x * numberIterations;

    // Continue if this index is smaller than the dimension of the instance.
    if(startIndexWithinInstance < numberEntriesPerInstance) {

        // What's the first entry index within the batch that this thread should operate on?
        int startIndexWithinBatch = blockIdx.x * numberEntriesPerInstance + startIndexWithinInstance;

        // Is the instance greater than the current batch size?
        if(blockIdx.x >= batchSize) {

            setToZero(destination, startIndexWithinBatch, numberIterations);

        }
        else {

            for(int indexEntry = startIndexWithinBatch; indexEntry < startIndexWithinBatch + numberIterations; indexEntry++) {

                destination[indexEntry] = relu(source[indexEntry]);

            }

        }

    }

}