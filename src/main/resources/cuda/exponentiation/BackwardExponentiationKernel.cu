extern "C"
__global__ void backwardExponentiationKernel (int length, float *forwardResults, float *chain, float *backwardResults)
{

    int globalId = blockDim.x * blockIdx.x + threadIdx.x;

    if(globalId < length) {

        backwardResults[globalId] = chain[globalId] * forwardResults[globalId];

    }

}