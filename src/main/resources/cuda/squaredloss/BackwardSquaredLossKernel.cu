extern "C"
__global__ void backwardSquaredLossKernel (int length, double *predictions, double *targets, double *result)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < length) {

        result[index] = predictions[index] - targets[index];

    }

    __syncthreads();

}