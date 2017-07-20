extern "C"
__global__ void exponentiationKernel (int length, double *source, double *destination)
{

    int globalId = blockDim.x * blockIdx.x + threadIdx.x;

    if(globalId < length) {

        destination[globalId] = exp(source[globalId]);

    }

}