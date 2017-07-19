extern "C"
__global__ void exponentiationKernel (double *source, double *destination)
{

    int globalId = blockDim.x * blockIdx.x + threadIdx.x;

    destination[globalId] = exp(source[globalId]);

}