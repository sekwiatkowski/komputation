#include "recurrent/BackwardRecurrent.cuh"

// first half of shared memory: differentiation w.r.t. pre-activation
// second half of shared memory: differentiation w.r.t. previous hidden state
__global__ void backwardRecurrentEmitAtEachStepKernel (
    int activationFunction,
    int entriesPerInstance,
    int hiddenDimension,
    int squaredHiddenDimension,
    int* lengths,
    int numberIterations,
    float* forwardResult,
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
    // There is no accumulator for the first state.
    int firstAccumulatorEntryIndex = firstInstanceEntry + (length-2) * squaredHiddenDimension;

    backwardLastStep(
        preActivation,
        forwardResult,
        chain,
        sharedData,
        backwardResult,
        firstStateEntryIndex,
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
            forwardResult,
            chain,
            sharedData,
            backwardResult,
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
        chain,
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