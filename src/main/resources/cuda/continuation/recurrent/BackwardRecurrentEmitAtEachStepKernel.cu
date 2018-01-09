#include "recurrent/BackwardRecurrent.cuh"
#include "arrays/copy/CopyCooperatively.cuh"
#include "arrays/add/AddCooperatively.cuh"

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

    // df(Wx+Uh+b)/d(Wx+Uh+b)
    // Differentiate activation w.r.t. pre-activation and write the result into the first half of shared memory
    backwardRecurrentActivation(preActivation, firstStateEntryIndex, chain, firstStateEntryIndex, sharedData, 0, startEntryIndex, exclusiveEndEntryIndex, activationFunction);

    copyCooperatively(sharedData, 0, backwardResult, firstStateEntryIndex, startEntryIndex, exclusiveEndEntryIndex);

    __syncthreads();

    // dUh/dh
    // Differentiate the weighted previous state w.r.t. the previous state and write the result into the second half of shared memory.
    // shared data is the differentiation w.r.t. the pre-activation
    backwardMatrixVectorMultiplicationWrtVector(previousStateWeights, sharedData, sharedData, hiddenDimension, startEntryIndex, exclusiveEndEntryIndex, hiddenDimension);

    int previousStateEntryIndex = firstStateEntryIndex - hiddenDimension;

    // dUh/dU
    // Differentiate the weighted previous state w.r.t. the weights and add to accumulator
    backwardMatrixVectorMultiplicationWrtMatrix(
        forwardResult,
        previousStateEntryIndex,
        sharedData,
        0,
        previousStateWeightAccumulation,
        firstAccumulatorEntryIndex,
        startEntryIndex,
        exclusiveEndEntryIndex,
        hiddenDimension);

    __syncthreads();

    for(int step = length - 2; step >= 1; step--) {

        firstStateEntryIndex -= hiddenDimension;
        firstAccumulatorEntryIndex -= squaredHiddenDimension;

        addCooperatively(chain, firstStateEntryIndex, sharedData, hiddenDimension, startEntryIndex, exclusiveEndEntryIndex);

        // Note that the differentiation w.r.t the previous state is in the second half of shared memory.
        backwardRecurrentActivation(preActivation, firstStateEntryIndex, sharedData, hiddenDimension, sharedData, 0, startEntryIndex, exclusiveEndEntryIndex, activationFunction);

        copyCooperatively(sharedData, 0, backwardResult, firstStateEntryIndex, startEntryIndex, exclusiveEndEntryIndex);

        __syncthreads();

        // Differentiate weighted previous state w.r.t previous state and write the result into the seconf half of shared memory.
        backwardMatrixVectorMultiplicationWrtVector(previousStateWeights, sharedData, sharedData, hiddenDimension, startEntryIndex, exclusiveEndEntryIndex, hiddenDimension);

        // Differentiate weighted previous state w.r.t weights and add to accumulator
        backwardMatrixVectorMultiplicationWrtMatrix(forwardResult, firstStateEntryIndex, sharedData, 0, previousStateWeightAccumulation, firstAccumulatorEntryIndex, startEntryIndex, exclusiveEndEntryIndex, hiddenDimension);

        __syncthreads();

    }

    firstStateEntryIndex = firstInstanceEntry;

    addCooperatively(chain, firstStateEntryIndex, sharedData, hiddenDimension, startEntryIndex, exclusiveEndEntryIndex);

    backwardRecurrentActivation(preActivation, firstStateEntryIndex, sharedData, hiddenDimension, backwardResult, firstStateEntryIndex, startEntryIndex, exclusiveEndEntryIndex, activationFunction);


    __syncthreads();

}