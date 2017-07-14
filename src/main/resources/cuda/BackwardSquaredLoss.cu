__device__ double backwardSquaredLoss (double prediction, double target)
{

    return prediction - target;

}

extern "C"
__global__ void backwardSquaredLossKernel (int length, double *predictions, double *targets, double *backwardResults)
{

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < length) {

        backwardResults[index] = backwardSquaredLoss(predictions[index], targets[index]);

    }

    __syncthreads();

}