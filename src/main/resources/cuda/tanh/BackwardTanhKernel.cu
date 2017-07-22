__device__ float backwardTanh (float forward, float chain)
{

    return chain * (1.0 - powf(forward, 2.0));

}

extern "C"
__global__ void backwardTanhKernel (int length, float *forward, float *chain, float *destination)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        destination[index] = backwardTanh(forward[index], chain[index]);

    }

}