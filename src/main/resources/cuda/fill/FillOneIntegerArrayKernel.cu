__global__ void fillOneIntegerArrayKernel(
    int numberRows,
    int numberEntries,
    int* array,
    int constant) {

    int index = blockIdx.x * numberEntries + blockIdx.y * numberRows + threadIdx.x;

    array[index] = constant;

}