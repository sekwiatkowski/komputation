__device__ float backwardSigmoid (float forward, float chain)
{

    return forward * (1.0f - forward) * chain;

}

extern "C"
__global__ void backwardSigmoidKernel (int numberEntriesPerInstance, float *forward, float *chain, float *destination)
{

    int indexInstance = blockIdx.x;
    int startInstance = indexInstance * numberEntriesPerInstance;
    int indexEntryInInstance = blockIdx.y * blockDim.y + threadIdx.x;
    int indexEntryInBatch = startInstance + indexEntryInInstance;

    if(indexEntryInInstance < numberEntriesPerInstance) {

        destination[indexEntryInBatch] = backwardSigmoid(forward[indexEntryInBatch], chain[indexEntryInBatch]);

    }

}