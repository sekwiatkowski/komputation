extern "C"
__global__ void accumulationKernel (int length, float *accumulator, float *addition)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        accumulator[index] += addition[index];

    }

}