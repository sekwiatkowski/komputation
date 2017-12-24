__global__ void fillOneIntegerArrayKernel(
    int numberEntries,
    int numberIterations,
    int* array,
    int constant) {
    int start = blockIdx.x * blockDim.x * numberIterations + threadIdx.x * numberIterations;

    for(int index = start; index < fminf(start + numberIterations, numberEntries); index++) {
        array[index] = constant;
    }
}