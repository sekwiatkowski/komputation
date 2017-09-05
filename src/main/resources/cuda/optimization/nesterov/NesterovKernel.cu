/*
    backup: v_backup = v
    history update: v = m * v - learning_rate * dx
    parameter update: x = x - m * v_backup + v + m * v
*/

__global__ void nesterovKernel (
    int numberIterations,
    float learningRate,
    float momentum,
    float* history,
    float* backup,
    int* parameterIndices,
    int* counts,
    int parameterSize,
    float* parameters,
    float* gradient) {

    int startEntry = (blockIdx.y * blockDim.x * numberIterations) + threadIdx.x * numberIterations;

    if(startEntry < parameterSize) {

        int gradientIndex = blockIdx.x;
        int parameterIndex = parameterIndices[gradientIndex];

        if(parameterIndex != -1) {

            int startParameter = parameterIndex * parameterSize + startEntry;
            int startGradient = gradientIndex * parameterSize + startEntry;

            float scalingFactor = 1.0 / (float)counts[gradientIndex];

            for(int indexParameter = startParameter, indexGradient = startGradient; indexParameter < startParameter + numberIterations; indexParameter++, indexGradient++) {

                float entryBackup = history[indexParameter];

                backup[indexParameter] = entryBackup;

                float entryUpdate = momentum * history[indexParameter] - scalingFactor * learningRate * gradient[indexGradient];

                history[indexParameter] = entryUpdate;

                float removedPreviousLookAhead = parameters[indexParameter] - momentum * entryBackup;

                parameters[indexParameter] = removedPreviousLookAhead + (1.0 + momentum) * entryUpdate;

            }

        }

    }

}