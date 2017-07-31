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
    float* gradient)
{

    int startEntry = threadIdx.x * numberIterations;

    if(startEntry < parameterSize) {

        int indexParameter = parameterIndices[blockIdx.x];

        int startGradient = blockIdx.x * parameterSize + startEntry;
        int startParameter = indexParameter * parameterSize + startEntry;

        for(int i = 0; i < numberIterations; i++) {

            float update = momentum * history[startParameter + i] - scalingFactor * learningRate * gradient[startGradient + i];
            history[startParameter + i] = update;

            parameters[startParameter + i] += update;

        }

    }

}