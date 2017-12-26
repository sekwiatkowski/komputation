__global__ void adamKernel (
    int numberIterations,
    int* hashTable,
    int* counts,
    int dimension,
    float* parameters,
    float* gradient,
    float learningRate,
    float firstMomentDecay,
    float oneMinusFirstMomentDecay,
    float secondMomentDecay,
    float oneMinusSecondMomentDecay,
    float epsilon,
    float step,
    float* firstMomentEstimate,
    float* secondMomentEstimate) {

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

                float updatedFirstMomentEstimate = firstMomentDecay * firstMomentEstimate[parameterEntryIndex] + oneMinusFirstMomentDecay * scaledDerivative;
                firstMomentEstimate[parameterEntryIndex] = updatedFirstMomentEstimate;
                float correctedFirstMomentEstimate = updatedFirstMomentEstimate / (1.0 - powf(firstMomentDecay, step));

                float updatedSecondMomentEstimate = secondMomentDecay * secondMomentEstimate[parameterEntryIndex] + oneMinusSecondMomentDecay * scaledDerivative * scaledDerivative;
                secondMomentEstimate[parameterEntryIndex] = updatedSecondMomentEstimate;
                float correctedSecondMomentEstimate = updatedSecondMomentEstimate / (1.0 - pow(secondMomentDecay, step));

                float adaptedLearningRate = learningRate / (sqrtf(correctedSecondMomentEstimate) + epsilon);

                float update = -correctedFirstMomentEstimate * adaptedLearningRate;

                parameters[parameterEntryIndex] += update;

                parameterEntryIndex++;
                gradientEntryIndex++;

            }

        }

    }

}