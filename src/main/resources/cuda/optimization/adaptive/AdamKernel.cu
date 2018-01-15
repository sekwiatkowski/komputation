#include "../../cuda.h"

__global__ void adamKernel (
    int numberIterations,
    int* parameterIndices,
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

    int updateIndex = blockIdx.x;
    int parameterIndex = parameterIndices[updateIndex];
    int count = counts[updateIndex];

    if(parameterIndex != -1 && count > 0) {

        float scalingFactor = 1.0f / (float)count;

        int startEntryIndex = (blockIdx.y * blockDim.x + threadIdx.x) * numberIterations;

        int firstParameterEntryIndex = parameterIndex * dimension;
        int startParameterEntryIndex = firstParameterEntryIndex + startEntryIndex;
        int startGradientEntryIndex = updateIndex * dimension + startEntryIndex;

        int exclusiveEndParameterEntryIndex = min(startParameterEntryIndex + numberIterations, firstParameterEntryIndex + dimension);

        int parameterEntryIndex = startParameterEntryIndex;
        int gradientEntryIndex = startGradientEntryIndex;

        while(parameterEntryIndex < exclusiveEndParameterEntryIndex) {

            float scaledDerivative = scalingFactor * gradient[gradientEntryIndex];

            float updatedFirstMomentEstimate = firstMomentDecay * firstMomentEstimate[parameterEntryIndex] + oneMinusFirstMomentDecay * scaledDerivative;
            firstMomentEstimate[parameterEntryIndex] = updatedFirstMomentEstimate;
            float correctedFirstMomentEstimate = updatedFirstMomentEstimate / (1.0f - powf(firstMomentDecay, step));

            float updatedSecondMomentEstimate = secondMomentDecay * secondMomentEstimate[parameterEntryIndex] + oneMinusSecondMomentDecay * scaledDerivative * scaledDerivative;
            secondMomentEstimate[parameterEntryIndex] = updatedSecondMomentEstimate;
            float correctedSecondMomentEstimate = updatedSecondMomentEstimate / (1.0f - powf(secondMomentDecay, step));

            float adaptedLearningRate = learningRate / (sqrtf(correctedSecondMomentEstimate) + epsilon);

            float update = -correctedFirstMomentEstimate * adaptedLearningRate;

            parameters[parameterEntryIndex] += update;

            parameterEntryIndex++;
            gradientEntryIndex++;

        }

    }

}