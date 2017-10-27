__global__ void adadeltaKernel (
    int numberIterations,
    int* parameterIndices,
    int* counts,
    int parameterSize,
    float* parameters,
    float* gradient,
    float decay,
    float oneMinusDecay,
    float epsilon,
    float* gradientAccumulation,
    float* updateAccumulation) {

    int startEntry = (blockIdx.y * blockDim.x * numberIterations) + threadIdx.x * numberIterations;

    if(startEntry < parameterSize) {

        int gradientIndex = blockIdx.x;
        int parameterIndex = parameterIndices[gradientIndex];

        if(parameterIndex != -1) {

            int startParameter = parameterIndex * parameterSize + startEntry;
            int startGradient = gradientIndex * parameterSize + startEntry;

            for(int indexParameter = startParameter, indexGradient = startGradient; indexParameter < startParameter + numberIterations; indexParameter++, indexGradient++) {

                float derivative = gradient[indexGradient];

                float newGradientAccumulation = decay * gradientAccumulation[parameterIndex] + oneMinusDecay * (derivative * derivative);
                gradientAccumulation[parameterIndex] = newGradientAccumulation;

                float rootMeanSquaredOfDerivatives = sqrtf(newGradientAccumulation + epsilon);

                float pastUpdateAccumulation = updateAccumulation[parameterIndex];
                float rootMeanSquaredOfPastUpdates = sqrtf(pastUpdateAccumulation + epsilon);

                float learningRate = rootMeanSquaredOfPastUpdates / rootMeanSquaredOfDerivatives;

                float update = -learningRate * derivative;

                updateAccumulation[parameterIndex] = decay * pastUpdateAccumulation + oneMinusDecay * (update * update);

                parameters[parameterIndex] += update;

            }

        }

    }

}