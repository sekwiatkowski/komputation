extern "C"
__global__ void backwardExponentiationKernel (double *forwardResults, double *chain, double *backwardResults)
{

    int globalId = blockDim.x * blockIdx.x + threadIdx.x;

    backwardResults[globalId] = chain[globalId] * forwardResults[globalId];

}