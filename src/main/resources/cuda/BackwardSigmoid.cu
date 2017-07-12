__device__ double backwardSigmoid (double forward, double chain)
{

    return forward * (1.0 - forward) * chain;

}

extern "C"
__global__ void backwardSigmoidKernel (int length, double *forward, double *chain, double *destination)
{

    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if(threadId < length) {

        destination[threadId] = backwardSigmoid(forward[threadId], chain[threadId]);

    }

}