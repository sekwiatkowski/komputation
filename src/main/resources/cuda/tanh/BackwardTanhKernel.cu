__device__ double backwardTanh (double forward, double chain)
{

    return chain * (1 - pow(forward, 2.0));

}

extern "C"
__global__ void backwardTanhKernel (int length, double *forward, double *chain, double *destination)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        destination[index] = backwardTanh(forward[index], chain[index]);

    }

}