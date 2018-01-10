#include "continuation/recurrent/Recurrent.cuh"
#include "symbols/NaN.cuh"

__global__ void recurrentLastStepKernel (
    int activationFunction,
    int maximumEntriesPerInstance,
    int hiddenDimension,
    int numberIterations,
    float* projectedInput,
    float* preActivation,
    float* previousStateWeights,
    int* lengths,
    int maximumLength,
    float* hiddenStates,
    float* result) {

    int instanceIndex = blockIdx.x;

    int firstInstanceEntryIndex = instanceIndex * maximumEntriesPerInstance;

    int startEntryIndex = threadIdx.x * numberIterations;
    // Do not go past the hidden dimension
    int exclusiveEndEntryIndex = min(startEntryIndex + numberIterations, hiddenDimension);

    extern __shared__ float sharedData[];

    forwardFirstStep(projectedInput, preActivation, hiddenStates, firstInstanceEntryIndex, sharedData, startEntryIndex, exclusiveEndEntryIndex, activationFunction);

    __syncthreads();

    int length = lengths[instanceIndex];

    int firstStateEntryIndex = firstInstanceEntryIndex;
    for(int step = 1; step < length; step++) {
        firstStateEntryIndex += hiddenDimension;

        forwardOtherStep(projectedInput, preActivation, hiddenStates, sharedData, previousStateWeights, firstStateEntryIndex, startEntryIndex, exclusiveEndEntryIndex, activationFunction, hiddenDimension);

        __syncthreads();
    }

    copyCooperatively(sharedData, 0, result, instanceIndex * hiddenDimension, startEntryIndex, exclusiveEndEntryIndex);

    for(int step = length; step < maximumLength; step++) {
        firstStateEntryIndex += hiddenDimension;

        setToNaN(result, firstStateEntryIndex + startEntryIndex, firstStateEntryIndex + exclusiveEndEntryIndex);
    }

}