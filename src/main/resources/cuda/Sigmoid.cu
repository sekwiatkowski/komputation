__device__ double sigmoid (double x)
{

    return 1.0 / (1.0 + exp (-x));

}

extern "C"
__global__ void sigmoidKernel (int length, double *source, double *destination)
{

    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        destination[index] = sigmoid(source[index]);

    }

    __syncthreads();

}