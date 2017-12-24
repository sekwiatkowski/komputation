__global__ void fillOneFloatArrayKernel(
    int numberEntries,
    int numberIterations,
    float* array,
    float constant) {
    int start = blockIdx.x * blockDim.x * numberIterations + threadIdx.x * numberIterations;

    for(int index = start; index < fminf(start + numberIterations, numberEntries); index++) {
        array[index] = constant;
    }
}