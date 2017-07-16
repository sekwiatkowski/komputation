extern "C"
__global__ void accumulationKernel (int length, double *accumulator, double *addition)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        accumulator[index] += addition[index];

    }

}