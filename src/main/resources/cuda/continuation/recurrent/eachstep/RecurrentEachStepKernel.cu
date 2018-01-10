#include "continuation/recurrent/Recurrent.cuh"
#include "symbols/NaN.cuh"

/*
    number of blocks in x dimension = number of instances
    number of threads = (hidden dimension +  maximum number of threads per block - 1) / maximum number of threads per block
    shared memory size = twice the state size

    This kernel assumes that the input has already been projected.
*/
__global__ void recurrentEachStepKernel (
    int activationFunction,
    int maximumEntriesPerInstance,
    int hiddenDimension,
    int numberIterations,
    float* projectedInput,
    float* preActivation,
    float* previousStateWeights,
    int *lengths,
    int maximumLength,
    float* hiddenStates) {

    int instanceIndex = blockIdx.x;

    int firstInstanceEntryIndex = instanceIndex * maximumEntriesPerInstance;

    int startEntryIndex = threadIdx.x * numberIterations;
    // Do not go past the hidden dimension
    int exclusiveEndEntryIndex = min(startEntryIndex + numberIterations, hiddenDimension);

    extern __shared__ float sharedData[];

    // computeFirstStep reads the input from the global memory and writes the output into shared memory.
    forwardFirstStep(projectedInput, preActivation, hiddenStates, firstInstanceEntryIndex, sharedData, startEntryIndex, exclusiveEndEntryIndex, activationFunction);

    __syncthreads();

    int length = lengths[instanceIndex];

    int firstStateEntryIndex = firstInstanceEntryIndex;
    for(int step = 1; step < length; step++) {
        firstStateEntryIndex += hiddenDimension;

        forwardOtherStep(projectedInput, preActivation, hiddenStates, sharedData, previousStateWeights, firstStateEntryIndex, startEntryIndex, exclusiveEndEntryIndex, activationFunction, hiddenDimension);

        __syncthreads();
    }

    for(int step = length; step < maximumLength; step++) {
        firstStateEntryIndex += hiddenDimension;

        setToNaN(hiddenStates, firstStateEntryIndex + startEntryIndex, firstStateEntryIndex + exclusiveEndEntryIndex);
    }


}