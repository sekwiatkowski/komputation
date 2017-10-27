__global__ void adamKernel (
    int numberIterations,
    int* parameterIndices,
    int* counts,
    int parameterSize,
    float* parameters,
    float* gradient,
    float learningRate,
    float firstMomentDecay,
    float oneMinusFirstMomentDecay,
    float secondMomentDecay,
    float oneMinusSecondMomentDecay,
    float epsilon,
    float step,
    float *firstMomentEstimate,
    float *secondMomentEstimate) {

    int startEntry = (blockIdx.y * blockDim.x * numberIterations) + threadIdx.x * numberIterations;

    if(startEntry < parameterSize) {

        int gradientIndex = blockIdx.x;
        int parameterIndex = parameterIndices[gradientIndex];

        if(parameterIndex != -1) {

            int startParameter = parameterIndex * parameterSize + startEntry;
            int startGradient = gradientIndex * parameterSize + startEntry;

            for(int indexParameter = startParameter, indexGradient = startGradient; indexParameter < startParameter + numberIterations; indexParameter++, indexGradient++) {

                float derivative = gradient[indexGradient];

                float updatedFirstMomentEstimate = firstMomentDecay * firstMomentEstimate[indexParameter] + oneMinusFirstMomentDecay * derivative;
                firstMomentEstimate[indexParameter] = updatedFirstMomentEstimate;
                float correctedFirstMomentEstimate = updatedFirstMomentEstimate / (1.0 - powf(firstMomentDecay, step));

                float updatedSecondMomentEstimate = secondMomentDecay * secondMomentEstimate[indexParameter] + oneMinusSecondMomentDecay * derivative * derivative;
                secondMomentEstimate[indexParameter] = updatedSecondMomentEstimate;
                float correctedSecondMomentEstimate = updatedSecondMomentEstimate / (1.0 - pow(secondMomentDecay, step));

                float adaptedLearningRate = learningRate / (sqrtf(correctedSecondMomentEstimate) + epsilon);

                float update = -correctedFirstMomentEstimate * adaptedLearningRate;

                parameters[indexParameter] += update;

            }

        }

    }

}