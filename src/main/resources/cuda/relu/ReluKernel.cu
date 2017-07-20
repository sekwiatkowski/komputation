__device__ double relu (double x)
{

    return fmax(x, 0.0);

}

extern "C"
__global__ void reluKernel (int length, double *source, double *destination)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        destination[index] = relu(source[index]);

    }

}