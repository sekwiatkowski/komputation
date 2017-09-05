__global__ void fillOneFloatArrayKernel(
    int numberRows,
    int numberEntries,
    float* array,
    float constant) {

    int index = blockIdx.x * numberEntries + blockIdx.y * numberRows + threadIdx.x;

    array[index] = constant;

}