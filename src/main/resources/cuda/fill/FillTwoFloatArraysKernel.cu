__global__ void fillTwoFloatsArraysKernel(
    int numberRows,
    int numberEntries,
    float* firstArray,
    float firstConstant,
    float* secondArray,
    float secondConstant) {

    int index = blockIdx.x * numberEntries + blockIdx.y * numberRows + threadIdx.x;

    firstArray[index] = firstConstant;
    secondArray[index] = secondConstant;

}