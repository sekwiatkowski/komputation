extern "C"
__global__ void backwardDropoutKernel (int numberEntries, float* chain, float* mask, float* result)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < numberEntries) {

        result[index] = chain[index] * mask[index];

    }

}