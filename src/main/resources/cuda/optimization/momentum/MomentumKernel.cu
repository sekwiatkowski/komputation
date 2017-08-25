extern "C"
__global__ void momentumKernel (
    int numberIterations,
    float learningRate,
    float momentum,
    float* history,
    int* parameterIndices,
    int parameterSize,
    float* parameters,
    float scalingFactor,
    float* gradient) {

    int startEntry = (blockIdx.y * blockDim.x * numberIterations) + threadIdx.x * numberIterations;

    if(startEntry < parameterSize) {

        int indexGradient = blockIdx.x;
        int indexParameter = parameterIndices[indexGradient];

        int startParameter = indexParameter * parameterSize + startEntry;
        int startGradient = indexGradient * parameterSize + startEntry;

        for(int indexParameter = startParameter, indexGradient = startGradient; indexParameter < startParameter + numberIterations; indexParameter++, indexGradient++) {

            float update = momentum * history[indexParameter] - scalingFactor * learningRate * gradient[indexGradient];

            history[indexParameter] = update;
            parameters[indexParameter] += update;

        }

    }

}