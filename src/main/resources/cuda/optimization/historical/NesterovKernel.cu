/*
    backup: v_backup = v
    history update: v = m * v - learning_rate * dx
    parameter update: x = x - m * v_backup + v + m * v
*/

__global__ void nesterovKernel (
    int numberIterations,
    int* hashTable,
    int* counts,
    int dimension,
    float* parameters,
    float* gradient,
    float learningRate,
    float momentum,
    float* history,
    float* backup) {
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
                float entryBackup = history[parameterEntryIndex];

                backup[parameterEntryIndex] = entryBackup;

                float scaledDerivative = scalingFactor * gradient[gradientEntryIndex];

                float entryUpdate = momentum * history[parameterEntryIndex] - learningRate * scaledDerivative;

                history[parameterEntryIndex] = entryUpdate;

                float removedPreviousLookAhead = parameters[parameterEntryIndex] - momentum * entryBackup;

                parameters[parameterEntryIndex] = removedPreviousLookAhead + (1.0 + momentum) * entryUpdate;

                parameterEntryIndex++;
                gradientEntryIndex++;
            }
        }
    }

}