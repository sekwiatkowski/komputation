__global__ void testtesttest(){
    printf("child");
}

__global__ void recurrentKernel (
    float *weightedInput,
    float *bias,
    float *result) {

    int indexEntry = threadIdx.x;

    result[indexEntry] = weightedInput[indexEntry] + bias[indexEntry];

    testtesttest<<<1,1>>>();

}