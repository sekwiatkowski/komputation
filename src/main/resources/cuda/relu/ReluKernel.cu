__device__ float relu (float x)
{

    return fmaxf(x, 0.0);

}

extern "C"
__global__ void reluKernel (int length, float *source, float *destination)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        destination[index] = relu(source[index]);

    }

}