extern "C"
__global__ void backwardSquaredLossKernel (int length, float *predictions, float *targets, float *result)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < length) {

        result[index] = predictions[index] - targets[index];

    }

}