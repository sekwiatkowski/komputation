__device__ double backwardSigmoid (double forward, double chain)
{

    return forward * (1.0 - forward) * chain;

}

extern "C"
__global__ void backwardSigmoidKernel (int length, double *forward, double *chain, double *destination)
{

    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        destination[index] = backwardSigmoid(forward[index], chain[index]);

    }

}