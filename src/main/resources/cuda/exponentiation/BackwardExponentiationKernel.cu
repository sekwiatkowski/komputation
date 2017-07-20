extern "C"
__global__ void backwardExponentiationKernel (int length, double *forwardResults, double *chain, double *backwardResults)
{

    int globalId = blockDim.x * blockIdx.x + threadIdx.x;

    if(globalId < length) {

        backwardResults[globalId] = chain[globalId] * forwardResults[globalId];

    }

}