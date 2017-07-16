__device__ double squaredLoss (double prediction, double target)
{

    return 0.5 * pow(prediction - target, 2.0);

}

extern "C"
__global__ void squaredLossKernel (int length, double *predictions, double *targets, double *forwardResults, double *result)
{

    extern __shared__ double sharedData[];

    // each thread loads one element from global to shared mem
    int threadId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    double forwardResult = squaredLoss(predictions[globalId], targets[globalId]);

    forwardResults[globalId] = forwardResult;
    sharedData[threadId] = forwardResult;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {

        if (threadId < stride) {

            sharedData[threadId] += sharedData[threadId + stride];

        }

        __syncthreads();
    }

    if (threadId == 0) {

        result[blockIdx.x] += sharedData[0];

    }

}