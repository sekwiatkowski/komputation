extern "C"
__global__ void exponentiationKernel (int length, float *source, float *destination)
{

    int globalId = blockDim.x * blockIdx.x + threadIdx.x;

    if(globalId < length) {

        destination[globalId] = expf(source[globalId]);

    }

}