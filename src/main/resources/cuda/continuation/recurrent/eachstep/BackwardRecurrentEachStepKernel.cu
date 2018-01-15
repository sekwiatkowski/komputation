#include "../../../cuda.h"
#include "../BackwardRecurrent.cuh"
#include "../../../arrays/add/AddCooperatively.cuh"

// first half of shared memory: differentiation w.r.t. pre-activation
// second half of shared memory: differentiation w.r.t. previous hidden state
__global__ void backwardRecurrentEachStepKernel (
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

    int firstInstanceEntryIndex = instanceIndex * entriesPerInstance;

    int startEntryIndex = threadIdx.x * numberIterations;
    int exclusiveEndEntryIndex = min(startEntryIndex + numberIterations, hiddenDimension);

    extern __shared__ float sharedData[];

    int length = lengths[instanceIndex];

    int firstStateEntryIndex = firstInstanceEntryIndex + (length-1) * hiddenDimension;
    int firstPreviousStateEntryIndex = firstStateEntryIndex - hiddenDimension;
    // There is no accumulator for the first state.
    int firstAccumulatorEntryIndex = firstInstanceEntryIndex + (length-2) * squaredHiddenDimension;

    backwardLastStep(
        preActivation,
        firstStateEntryIndex,
        firstPreviousStateEntryIndex,
        hiddenStates,
        backwardResult,
        chain,
        firstStateEntryIndex,
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

        addCooperatively(
            chain,
            firstStateEntryIndex,
            sharedData,
            hiddenDimension,
            startEntryIndex,
            exclusiveEndEntryIndex);

        backwardStepsInBetween(
            preActivation,
            hiddenStates,
            backwardResult,
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

    addCooperatively(
        chain,
        firstInstanceEntryIndex,
        sharedData,
        hiddenDimension,
        startEntryIndex,
        exclusiveEndEntryIndex);

    backwardFirstStep(
        preActivation,
        backwardResult,
        firstInstanceEntryIndex,
        sharedData,
        hiddenDimension,
        startEntryIndex,
        exclusiveEndEntryIndex,
        activationFunction
    );


}