__global__ void momentumKernel (
    int numberIterations,
    int* hashTable,
    int* counts,
    int dimension,
    float* parameters,
    float* gradient,
    float learningRate,
    float momentum,
    float* history) {
    int firstEntryIndex = (blockIdx.y * blockDim.x * numberIterations) + threadIdx.x * numberIterations;

    if(firstEntryIndex < dimension) {
        int hashTableIndex = blockIdx.x;
        int parameterIndex = hashTable[hashTableIndex];

        if(parameterIndex != -1) {
            int parameterIndex = hashTable[hashTableIndex];

            int count = counts[hashTableIndex];
            float scalingFactor = 1.0 / (float)count;

            int firstParameterEntryIndex = parameterIndex * dimension + firstEntryIndex;
            int firstGradientEntryIndex = hashTableIndex * dimension + firstEntryIndex;

            int exclusiveLastParameterEntryIndex = firstParameterEntryIndex + numberIterations;

            int parameterEntryIndex = firstParameterEntryIndex;
            int gradientEntryIndex = firstGradientEntryIndex;

            while(parameterEntryIndex < exclusiveLastParameterEntryIndex) {
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

}