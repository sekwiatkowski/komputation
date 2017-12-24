__global__ void groupSumKernel(
    int dimension,
    int parametersPerInstance,
    int* mapping,
    float* gradient,
    float* groupSum) {
    int indexInstance = blockIdx.x;
    int indexParameterWithinInstance = blockIdx.y;

    int indexParameterWithinBatch = indexInstance * parametersPerInstance + indexParameterWithinInstance;

    int slot = mapping[indexParameterWithinBatch];

    if(slot != -1) {
        int indexEntry = threadIdx.x;

        int groupSumEntryIndex = slot * dimension + indexEntry;
        int gradientEntryIndex = indexParameterWithinBatch * dimension + indexEntry;

        atomicAdd(&groupSum[groupSumEntryIndex], gradient[gradientEntryIndex]);
    }
}