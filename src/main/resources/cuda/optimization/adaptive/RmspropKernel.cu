__global__ void rmspropKernel (
    int numberIterations,
    int* parameterIndices,
    int* counts,
    int parameterSize,
    float* parameters,
    float* gradient,
    float learningRate,
    float decay,
    float oneMinusDecay,
    float epsilon,
    float* accumulation) {

    int startEntry = (blockIdx.y * blockDim.x * numberIterations) + threadIdx.x * numberIterations;

    if(startEntry < parameterSize) {

        int gradientIndex = blockIdx.x;
        int parameterIndex = parameterIndices[gradientIndex];

        if(parameterIndex != -1) {

            int startParameter = parameterIndex * parameterSize + startEntry;
            int startGradient = gradientIndex * parameterSize + startEntry;

            for(int indexParameter = startParameter, indexGradient = startGradient; indexParameter < startParameter + numberIterations; indexParameter++, indexGradient++) {

                float derivative = gradient[indexGradient];

                float updatedAccumulation = decay * accumulation[indexParameter] + oneMinusDecay * (derivative * derivative);
                accumulation[indexParameter] = updatedAccumulation;

                float adaptiveLearningRate = learningRate / sqrtf(updatedAccumulation + epsilon);
                float update = -adaptiveLearningRate * derivative;

                parameters[indexParameter] += update;

            }

        }

    }

}