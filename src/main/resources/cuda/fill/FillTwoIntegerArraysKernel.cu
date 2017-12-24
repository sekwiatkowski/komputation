__global__ void fillTwoIntegerArraysKernel(
    int numberEntries,
    int numberIterations,
    int* firstArray,
    int firstConstant,
    int* secondArray,
    int secondConstant) {
    int start = blockIdx.x * blockDim.x * numberIterations + threadIdx.x * numberIterations;

    for(int index = start; index < fminf(start + numberIterations, numberEntries); index++) {
        firstArray[index] = firstConstant;
        secondArray[index] = secondConstant;
    }
}