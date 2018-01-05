__global__ void momentumKernel (
    int numberIterations,
    int* parameterIndices,
    int* counts,
    int dimension,
    float* parameters,
    float* gradient,
    float learningRate,
    float momentum,
    float* history) {

    int updateIndex = blockIdx.x;
    int parameterIndex = parameterIndices[updateIndex];
    int count = counts[updateIndex];

    if(parameterIndex != -1 && count > 0) {
        float scalingFactor = 1.0 / (float)count;

        int startEntryIndex = (blockIdx.y * blockDim.x + threadIdx.x) * numberIterations;

        int firstParameterEntryIndex = parameterIndex * dimension;
        int startParameterEntryIndex = firstParameterEntryIndex + startEntryIndex;
        int startGradientEntryIndex = updateIndex * dimension + startEntryIndex;

        int exclusiveEndParameterEntryIndex = min(startParameterEntryIndex + numberIterations, firstParameterEntryIndex + dimension);

        int parameterEntryIndex = startParameterEntryIndex;
        int gradientEntryIndex = startGradientEntryIndex;

        while(parameterEntryIndex < exclusiveEndParameterEntryIndex) {
            float derivative = gradient[gradientEntryIndex];
            float scaledDerivative = scalingFactor * derivative;

            float update = momentum * history[parameterEntryIndex] - learningRate * scaledDerivative;

            history[parameterEntryIndex] = update;
            parameters[parameterEntryIndex] += update;

            parameterEntryIndex++;
            gradientEntryIndex++;
        }
    }

}