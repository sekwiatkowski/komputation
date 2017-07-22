__device__ float backwardSigmoid (float forward, float chain)
{

    return forward * (1.0 - forward) * chain;

}

extern "C"
__global__ void backwardSigmoidKernel (int length, float *forward, float *chain, float *destination)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        destination[index] = backwardSigmoid(forward[index], chain[index]);

    }

}