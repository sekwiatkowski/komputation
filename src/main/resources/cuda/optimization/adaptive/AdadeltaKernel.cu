__global__ void adadeltaKernel (
    int numberIterations,
    int* hashTable,
    int* counts,
    int dimension,
    float* parameters,
    float* gradient,
    float decay,
    float oneMinusDecay,
    float epsilon,
    float* gradientAccumulation,
    float* updateAccumulation) {

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

                float newGradientAccumulation = decay * gradientAccumulation[parameterEntryIndex] + oneMinusDecay * (scaledDerivative * scaledDerivative);
                gradientAccumulation[parameterEntryIndex] = newGradientAccumulation;

                float rootMeanSquaredOfDerivatives = sqrtf(newGradientAccumulation + epsilon);

                float pastUpdateAccumulation = updateAccumulation[parameterEntryIndex];
                float rootMeanSquaredOfPastUpdates = sqrtf(pastUpdateAccumulation + epsilon);

                float learningRate = rootMeanSquaredOfPastUpdates / rootMeanSquaredOfDerivatives;

                float update = -learningRate * scaledDerivative;

                updateAccumulation[parameterEntryIndex] = decay * pastUpdateAccumulation + oneMinusDecay * (update * update);

                parameters[parameterEntryIndex] += update;

                parameterEntryIndex++;
                gradientEntryIndex++;
            }
        }
    }

}