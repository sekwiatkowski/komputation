__global__ void fillTwoIntegerArraysKernel(
    int numberRows,
    int numberEntries,
    int* firstArray,
    int firstConstant,
    int* secondArray,
    int secondConstant) {

    int index = blockIdx.x * numberEntries + blockIdx.y * numberRows + threadIdx.x;

    firstArray[index] = firstConstant;
    secondArray[index] = secondConstant;

}