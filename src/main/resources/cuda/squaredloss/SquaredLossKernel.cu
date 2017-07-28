#include "reduction/Reduction.cuh"

// This assumes that the number of threads is equal to the number of predictions/targets.
// First, the squared differences between predictions and targets are stored in shared memory.
// In the second step, the squared differences are summed up using a parallel reduction.
// Finally, the sum is multiplied by 1/2.
template <int blockSize>
__global__ void squaredLossKernel (int numberEntries, float *predictions, float *targets, float *result)
{

    int threadId = threadIdx.x;

    extern __shared__ float sharedData[];

    if(threadId < numberEntries) {

        sharedData[threadId] = powf(predictions[threadId] - targets[threadId], 2.0);

    }

    __syncthreads();

    reduce<blockSize>(threadId, sharedData, 0);

    if(threadId == 0) {

        result[0] = 0.5 * sharedData[0];

    }

}