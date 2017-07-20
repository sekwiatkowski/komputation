__device__ double tanh (double x)
{

    return (2.0 / (1.0 + exp(-2.0*x))) - 1.0;

}

extern "C"
__global__ void tanhKernel (int length, double *source, double *destination)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        destination[index] = tanh(source[index]);

    }

}