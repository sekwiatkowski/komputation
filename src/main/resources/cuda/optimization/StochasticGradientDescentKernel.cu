__global__ void stochasticGradientDescentKernel (
    int numberIterations,
    int* hashTable,
    int* counts,
    int dimension,
    float* parameters,
    float* gradient,
    float learningRate) {
    int firstEntryIndex = (blockIdx.y * blockDim.x * numberIterations) + threadIdx.x * numberIterations;

    if(firstEntryIndex < dimension) {

        int hashTableIndex = blockIdx.x;
        int parameterIndex = hashTable[hashTableIndex];

        if(parameterIndex != -1) {
            float scalingFactor = 1.0 / (float)counts[hashTableIndex];

            int firstParameterEntryIndex = parameterIndex * dimension + firstEntryIndex;
            int firstGradientEntryIndex = hashTableIndex * dimension + firstEntryIndex;

            int exclusiveLastParameterEntryIndex = firstParameterEntryIndex + numberIterations;

            int parameterEntryIndex = firstParameterEntryIndex;
            int gradientEntryIndex = firstGradientEntryIndex;

            while(parameterEntryIndex < exclusiveLastParameterEntryIndex) {
                float scaledDerivative = scalingFactor * gradient[gradientEntryIndex];

                parameters[parameterEntryIndex] -= learningRate * scaledDerivative;

                parameterEntryIndex++;
                gradientEntryIndex++;
            }
        }
    }
}