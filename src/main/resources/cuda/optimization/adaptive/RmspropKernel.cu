__global__ void rmspropKernel (
    int numberIterations,
    int* hashTable,
    int* counts,
    int dimension,
    float* parameters,
    float* gradient,
    float learningRate,
    float decay,
    float oneMinusDecay,
    float epsilon,
    float* accumulation) {

    int firstEntryIndex = (blockIdx.y * blockDim.x * numberIterations) + threadIdx.x * numberIterations;

    if(firstEntryIndex < dimension) {
        int hashTableIndex = blockIdx.x;
        int parameterIndex = hashTable[hashTableIndex];

        if(parameterIndex != -1) {
            int parameterIndex = hashTable[hashTableIndex];

            float scalingFactor = 1.0 / (float)counts[hashTableIndex];

            int firstParameterEntryIndex = parameterIndex * dimension + firstEntryIndex;
            int firstGradientEntryIndex = hashTableIndex * dimension + firstEntryIndex;

            int exclusiveLastParameterEntryIndex = firstParameterEntryIndex + numberIterations;

            int parameterEntryIndex = firstParameterEntryIndex;
            int gradientEntryIndex = firstGradientEntryIndex;

            while(parameterEntryIndex < exclusiveLastParameterEntryIndex) {
                float scaledDerivative = scalingFactor * gradient[gradientEntryIndex];

                float updatedAccumulation = decay * accumulation[parameterEntryIndex] + oneMinusDecay * (scaledDerivative * scaledDerivative);
                accumulation[parameterEntryIndex] = updatedAccumulation;

                float adaptiveLearningRate = learningRate / sqrtf(updatedAccumulation + epsilon);
                float update = -adaptiveLearningRate * scaledDerivative;

                parameters[parameterEntryIndex] += update;

                parameterEntryIndex++;
                gradientEntryIndex++;
            }

        }

    }

}