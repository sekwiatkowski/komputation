__device__ float sigmoid (float x)
{

    return 1.0 / (1.0 + expf (-x));

}

extern "C"
__global__ void sigmoidKernel (int length, float *source, float *destination)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        destination[index] = sigmoid(source[index]);

    }

}