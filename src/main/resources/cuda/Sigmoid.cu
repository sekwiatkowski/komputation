__device__ double sigmoid (double x)
{

    return 1.0 / (1.0 + exp (-x));

}

extern "C"
__global__ void sigmoidKernel (int length, double *source, double *destination)
{

    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if(threadId < length) {

        destination[threadId] = sigmoid(source[threadId]);

    }

}