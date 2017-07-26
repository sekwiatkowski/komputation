extern "C"
__global__ void dropoutRuntimeKernel (int numberEntries, float keepProbability, float* input, float* result)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < numberEntries) {

        result[index] = keepProbability * input[index];

    }

}