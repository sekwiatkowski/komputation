/*
    backup: v_backup = v
    history update: v = m * v - learning_rate * dx
    parameter update: x = x - m * v_backup + v + m * v
*/

__global__ void nesterovKernel (
    int numberIterations,
    int* parameterIndices,
    int* counts,
    int dimension,
    float* parameters,
    float* gradient,
    float learningRate,
    float momentum,
    float* history,
    float* backup) {

    int updateIndex = blockIdx.x;
    int parameterIndex = parameterIndices[updateIndex];
    int count = counts[updateIndex];

    if(parameterIndex != -1 && count > 0) {
        float scalingFactor = 1.0 / (float)count;

        int startEntryIndex = (blockIdx.y * blockDim.x + threadIdx.x) * numberIterations;

        int firstParameterEntryIndex = parameterIndex * dimension;
        int startParameterEntryIndex = firstParameterEntryIndex + startEntryIndex;
        int exclusiveEndParameterEntryIndex = min(startParameterEntryIndex + numberIterations, firstParameterEntryIndex + dimension);

        int startGradientEntryIndex = updateIndex * dimension + startEntryIndex;

        int parameterEntryIndex = startParameterEntryIndex;
        int gradientEntryIndex = startGradientEntryIndex;

        while(parameterEntryIndex < exclusiveEndParameterEntryIndex) {
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