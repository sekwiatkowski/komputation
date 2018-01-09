/*
    Number of threads = (batch size * entries per instance + maximum threads per block - 1) / maximum threads per block
 */
__global__ void sumKernel(
    float* input,
    float* result,
    int batchSize,
    int numberEntries,
    int numberIterations) {

    int startEntry = (blockIdx.x * blockDim.x + threadIdx.x) * numberIterations;
    int exclusiveEndEntry = min(startEntry + numberIterations, numberEntries);

    for(int entryIndex = startEntry; entryIndex < exclusiveEndEntry; entryIndex++) {

        float entry = 0.0;

        for(int instanceIndex = 0; instanceIndex < batchSize; instanceIndex++) {
            entry += input[instanceIndex * numberEntries + entryIndex];
        }

        result[entryIndex] = entry;
    }

}