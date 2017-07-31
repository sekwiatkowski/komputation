#include "zero/Zero.cuh"

__device__ float backwardRelu (float forward, float chain)
{

    if(forward > 0.0) {

        return chain;

    }
    else {

        return 0.0;

    }

}

extern "C"
__global__ void backwardReluKernel (int batchSize, int numberEntriesPerInstance, int numberIterations, float *forward, float *chain, float *destination) {

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

                destination[indexEntry] = backwardRelu(forward[indexEntry], chain[indexEntry]);

            }

        }

    }

}