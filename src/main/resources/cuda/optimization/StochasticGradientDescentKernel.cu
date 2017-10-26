__global__ void stochasticGradientDescentKernel (
    int numberIterations,
    int* parameterIndices,
    int* counts,
    int parameterSize,
    float* parameters,
    float* gradient,
    float learningRate,) {

    int startEntry = (blockIdx.y * blockDim.x * numberIterations) + threadIdx.x * numberIterations;

    if(startEntry < parameterSize) {

        int gradientIndex = blockIdx.x;
        int parameterIndex = parameterIndices[gradientIndex];

        if(parameterIndex != -1) {

            int startParameter = parameterIndex * parameterSize + startEntry;
            int startGradient = gradientIndex * parameterSize + startEntry;

            float scalingFactor = 1.0 / (float)counts[gradientIndex];

            for(int indexParameter = startParameter, indexGradient = startGradient; indexParameter < startParameter + numberIterations; indexParameter++, indexGradient++) {

                parameters[indexParameter] -= scalingFactor * learningRate * gradient[indexGradient];

            }

        }

    }

}