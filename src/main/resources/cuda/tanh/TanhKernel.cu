__device__ float tanh (float x)
{

    return (2.0 / (1.0 + expf(-2.0*x))) - 1.0;

}

extern "C"
__global__ void tanhKernel (int length, float *source, float *destination)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        destination[index] = tanh(source[index]);

    }

}