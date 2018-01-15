#include "../../cuda.h"

__global__ void rmspropKernel (
    int numberIterations,
    int* parameterIndices,
    int* counts,
    int dimension,
    float* parameters,
    float* gradient,
    float learningRate,
    float decay,
    float oneMinusDecay,
    float epsilon,
    float* accumulation) {

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

            float updatedAccumulation = decay * accumulation[parameterEntryIndex] + oneMinusDecay * (scaledDerivative * scaledDerivative);
            accumulation[parameterEntryIndex] = updatedAccumulation;

            float adaptiveLearningRate = learningRate / sqrtf(updatedAccumulation + epsilon);
            float update = -adaptiveLearningRate * scaledDerivative;

            parameters[parameterEntryIndex] += update;

            parameterEntryIndex++;
            gradientEntryIndex++;
        }

    }

}