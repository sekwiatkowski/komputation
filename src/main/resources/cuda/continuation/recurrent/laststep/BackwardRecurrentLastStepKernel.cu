#include "continuation/recurrent/BackwardRecurrent.cuh"

// first half of shared memory: differentiation w.r.t. pre-activation
// second half of shared memory: differentiation w.r.t. previous hidden state
__global__ void backwardRecurrentLastStepKernel (
    int activationFunction,
    int entriesPerInstance,
    int hiddenDimension,
    int squaredHiddenDimension,
    int* lengths,
    int numberIterations,
    float* hiddenStates,
    float* preActivation,
    float* previousStateWeights,
    float* previousStateWeightAccumulation,
    float* chain,
    float* backwardResult) {

    int instanceIndex = blockIdx.x;

    int firstInstanceEntry = instanceIndex * entriesPerInstance;

    int startEntryIndex = threadIdx.x * numberIterations;
    int exclusiveEndEntryIndex = min(startEntryIndex + numberIterations, hiddenDimension);

    extern __shared__ float sharedData[];

    int length = lengths[instanceIndex];

    int firstStateEntryIndex = firstInstanceEntry + (length-1) * hiddenDimension;
    int firstPreviousStateEntryIndex = firstStateEntryIndex - hiddenDimension;
    int firstResultIndex = instanceIndex * hiddenDimension;
    // There is no accumulator for the first state.
    int firstAccumulatorEntryIndex = firstInstanceEntry + (length-2) * squaredHiddenDimension;

    backwardLastStep(
        preActivation,
        firstStateEntryIndex,
        firstPreviousStateEntryIndex,
        hiddenStates,
        backwardResult,
        chain,
        firstResultIndex,
        sharedData,
        previousStateWeights,
        previousStateWeightAccumulation,
        firstAccumulatorEntryIndex,
        startEntryIndex,
        exclusiveEndEntryIndex,
        hiddenDimension,
        activationFunction);

    __syncthreads();

    for(int step = length - 2; step >= 1; step--) {

        firstStateEntryIndex -= hiddenDimension;
        firstAccumulatorEntryIndex -= squaredHiddenDimension;

        backwardStepsInBetween(
            preActivation,
            hiddenStates,
            backwardResult,
            chain,
            sharedData,
            firstStateEntryIndex,
            sharedData,
            hiddenDimension,
            previousStateWeights,
            previousStateWeightAccumulation,
            firstAccumulatorEntryIndex,
            startEntryIndex,
            exclusiveEndEntryIndex,
            hiddenDimension,
            activationFunction
        );

        __syncthreads();

    }

    backwardFirstStep(
        preActivation,
        backwardResult,
        firstInstanceEntry,
        sharedData,
        hiddenDimension,
        startEntryIndex,
        exclusiveEndEntryIndex,
        hiddenDimension,
        activationFunction
    );

}