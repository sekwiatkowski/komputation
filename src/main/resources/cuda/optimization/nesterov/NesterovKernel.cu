/*
    backup: v_backup = v
    history update: v = m * v - learning_rate * dx
    parameter update: x = x - m * v_backup + v + m * v
*/

extern "C"
__global__ void nesterovKernel (
    int numberIterations,
    float learningRate,
    float momentum,
    float* history,
    float* backup,
    int* parameterIndices,
    int parameterSize,
    float* parameters,
    float scalingFactor,
    float* gradient) {

    int startEntry = (blockIdx.y * blockDim.x * numberIterations) + threadIdx.x * numberIterations;

    if(startEntry < parameterSize) {

        int indexGradient = blockIdx.x;
        int indexParameter = parameterIndices[indexGradient];

        int startParameter = indexParameter * parameterSize + startEntry;
        int startGradient = indexGradient * parameterSize + startEntry;

        for(int i = 0; i < numberIterations; i++) {

            int indexEntry = startParameter + i;

            float entryBackup = history[indexEntry];

            backup[indexEntry] = entryBackup;

            float entryUpdate = momentum * history[indexEntry] - scalingFactor * learningRate * gradient[startGradient + i];

            history[indexEntry] = entryUpdate;

            float removedPreviousLookAhead = parameters[indexEntry] - momentum * entryBackup;

            parameters[indexEntry] = removedPreviousLookAhead + (1.0f + momentum) * entryUpdate;

        }

    }

}