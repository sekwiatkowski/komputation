#include "../../cuda.h"

__global__ void adagradKernel (
    int numberIterations,
    int* parameterIndices,
    int* counts,
    int dimension,
    float* parameters,
    float* gradient,
    float learningRate,
    float* history,
    float epsilon) {

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

            float updatedHistory = history[parameterEntryIndex] + scaledDerivative * scaledDerivative;

            history[parameterEntryIndex] = updatedHistory;

            float adaptedLearningRate = learningRate / (sqrtf(updatedHistory) + epsilon);

            float update = adaptedLearningRate * scalingFactor * gradient[gradientEntryIndex];

            parameters[parameterEntryIndex] -= update;

            parameterEntryIndex++;
            gradientEntryIndex++;
        }
    }
}