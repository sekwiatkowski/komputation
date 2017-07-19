__global__ void backwardNormalizationKernel (double * inputs, double * sums, double * results)
{

    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int globalId = blockId * blockDim.x + threadId;

    extern __shared__ double sharedData[];

    if(threadId == 0) {

        sharedData[0] = sums[blockId] * sums[blockId];

    }

    __syncthreads();

    results[globalId] = (sums[blockId] - inputs[globalId]) / sharedData[0];

}