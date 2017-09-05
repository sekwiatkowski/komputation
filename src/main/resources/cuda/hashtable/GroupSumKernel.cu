__global__ void groupSumKernel(
    int dimension,
    int maximumKeys,
    int* mapping,
    float* gradient,
    float* groupSum) {

    int indexInstance = blockIdx.x;
    int indexKey = blockIdx.y;

    int indexMapping = indexInstance * maximumKeys + indexKey;
    int slot = mapping[indexMapping];

    if(slot != -1) {

        int indexEntry = threadIdx.x;
        int indexGroupSumEntry = slot * dimension + indexEntry;

        int indexGradientEntry = indexInstance * maximumKeys * dimension + indexKey * dimension + indexEntry;
        float gradientEntry = gradient[indexGradientEntry];

        atomicAdd(&groupSum[indexGroupSumEntry], gradientEntry);

    }

}