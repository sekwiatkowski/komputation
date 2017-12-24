__global__ void fillTwoFloatsArraysKernel(
    int numberRows,
    int numberEntries,
    float* firstArray,
    float firstConstant,
    float* secondArray,
    float secondConstant) {
    int start = indexInstance * numberEntries + indexColumn * numberRows + threadIdx.x;

    for(int index = start; index < fminf(start + numberIterations, numberEntries); index++) {
        firstArray[indexEntry] = firstConstant;
        secondArray[indexEntry] = secondConstant;
    }
}