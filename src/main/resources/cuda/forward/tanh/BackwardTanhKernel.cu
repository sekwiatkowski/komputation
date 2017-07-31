#include "zero/Zero.cuh"

__device__ float backwardTanh (float forward, float chain)
{

    return chain * (1.0 - powf(forward, 2.0));

}

extern "C"
__global__ void backwardTanhKernel (int batchSize, int numberEntriesPerInstance, int numberIterations, float *forward, float *chain, float *destination) {

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

                destination[indexEntry] = backwardTanh(forward[indexEntry], chain[indexEntry]);

            }

        }

    }

}