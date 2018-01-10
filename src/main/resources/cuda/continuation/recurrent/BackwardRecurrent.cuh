#include "continuation/recurrent/RecurrentActivation.cuh"
#include "continuation/relu/Relu.cuh"
#include "continuation/sigmoid/Sigmoid.cuh"
#include "continuation/tanh/Tanh.cuh"
#include "arrays/copy/CopyCooperatively.cuh"
#include "arrays/add/AddCooperatively.cuh"

__device__ void backwardRecurrentActivation(
    float* input,
    int startInput,
    float* chain,
    int startChain,
    float* result,
    int startResult,
    int startEntryIndex,
    int exclusiveEndEntryIndex,
    int activationFunction) {

    switch(activationFunction) {
        case IDENTITY:
            for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
                result[startResult + entryIndex] = chain[startChain + entryIndex];
            }
            break;
        case RELU:
            for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
                result[startResult + entryIndex] = backwardRelu(input[startInput + entryIndex], chain[startChain + entryIndex]);
            }
            break;
        case SIGMOID:
            for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
                result[startResult + entryIndex] = backwardSigmoid(input[startInput + entryIndex], chain[startChain + entryIndex]);
            }
            break;
        case TANH:
            for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
                result[startResult + entryIndex] = backwardTanh(input[startInput + entryIndex], chain[startChain + entryIndex]);
            }
            break;
    }
}

/*
    w_11 * x_1 + w_12 * x_2 + ... + w_1n * x_n
    w_21 * x_1 + w_22 * x_2 + ... + w_2n * x_n
    ...
    w_n1 * x_1 + w_n2 * x_2 + ... + w_nn * x_n

    dWx/dx_1: w_11 + w_21 + ... + w_n1
    dWx/dx_2: w_12 + w_22 + ... + w_n2
    ...
    dWx/dx_n: w_1n + w_2n + ... + w_nn

                       c_1
                       c_2
                       ...
                       c_n
    w_11 w_21 ... w_n1
    w_11 w_21 ... w_n1

*/
__device__ void backwardMatrixVectorMultiplicationWrtVector(float* matrix, float* chain, float* result, int startResult, int startEntryIndex, int exclusiveEndEntryIndex, int dimension) {

    // Matrix row == result index
    for(int matrixRow = startEntryIndex; matrixRow < exclusiveEndEntryIndex; matrixRow++) {

        float entryResult = 0.0;

        for(int matrixColumn = 0; matrixColumn < dimension; matrixColumn++) {
            // Note that the weight matrix is transposed!
            entryResult += matrix[matrixRow * dimension + matrixColumn] * chain[matrixRow];
        }

        result[startResult + matrixRow] = entryResult;

    }

}

/*
    w_11 * x_1 + w_12 * x_2 + ... + w_1n * x_n
    w_21 * x_1 + w_22 * x_2 + ... + w_2n * x_n
    ...
    w_n1 * x_1 + w_n2 * x_2 + ... + w_nn * x_n

    dWx/dw_11: x_1 dWx/dw_12: x_2 ... dWx/dw_12: x_n
    dWx/dw_21: x_1 dWx/dw_22: x_2 ... dWx/dw_2n: x_n
    ...
    dWx/dw_n1: x_1 dWx/dw_n2: x_2 ... dWx/dw_nn: x_n

        x_1, x_2, ..., x_n
    c_1
    c_2
    ...
    c_n

*/
__device__ void backwardMatrixVectorMultiplicationWrtMatrix(
    float* vector,
    int firstInput,
    float* chain,
    int firstChain,
    float* result,
    int firstResult,
    int startEntryIndex,
    int exclusiveEndEntryIndex,
    int dimension) {

    for(int matrixRow = startEntryIndex; matrixRow < exclusiveEndEntryIndex; matrixRow++) {
        float chainEntry = chain[firstChain + matrixRow];

        int firstResultRow = firstResult + matrixRow;

        for(int matrixColumn = 0; matrixColumn < dimension; matrixColumn++) {
            result[firstResultRow + matrixColumn * dimension] = chainEntry * vector[firstInput + matrixColumn];
        }
    }

}

__device__ void backwardLastStep(
    float* preActivation,
    int firstStateEntryIndex,
    int firstPreviousStateEntryIndex,
    float* hiddenStates,
    float* backwardResult,
    float* chain,
    int firstResultEntryIndex,
    float* sharedMemory,
    float* previousStateWeights,
    float* previousStateWeightAccumulation,
    int firstAccumulatorEntryIndex,
    int startEntryIndex,
    int exclusiveEndEntryIndex,
    int hiddenDimension,
    int activationFunction) {

    // df(Wx+Uh+b)/d(Wx+Uh+b)
    // Differentiate activation w.r.t. pre-activation and write the result into the first half of shared memory.
    backwardRecurrentActivation(
        preActivation,
        firstStateEntryIndex,
        chain,
        firstResultEntryIndex,
        sharedMemory,
        0,
        startEntryIndex,
        exclusiveEndEntryIndex,
        activationFunction);

    __syncthreads();

    // dUh_t-1/dh_t-1
    // Differentiate the weighted previous state w.r.t. the previous state and write the result into the second half of shared memory.
    backwardMatrixVectorMultiplicationWrtVector(
        previousStateWeights,
        sharedMemory,
        sharedMemory,
        hiddenDimension,
        startEntryIndex,
        exclusiveEndEntryIndex,
        hiddenDimension);

    // dUh/dU
    // Differentiate the weighted previous state w.r.t. the weights and add to accumulator
    backwardMatrixVectorMultiplicationWrtMatrix(
        hiddenStates,
        firstPreviousStateEntryIndex,
        sharedMemory,
        0,
        previousStateWeightAccumulation,
        firstAccumulatorEntryIndex,
        startEntryIndex,
        exclusiveEndEntryIndex,
        hiddenDimension);

    copyCooperatively(
        sharedMemory,
        0,
        backwardResult,
        firstStateEntryIndex,
        startEntryIndex,
        exclusiveEndEntryIndex);

}

__device__ void backwardFirstStep(
    float* preActivation,
    float* backwardResult,
    int firstStateEntryIndex,
    float* sumOfChainAndBackwardPreviousState,
    int firstSumOfChainAndBackwardPreviousState,
    int startEntryIndex,
    int exclusiveEndEntryIndex,
    int hiddenDimension,
    int activationFunction) {

    backwardRecurrentActivation(
        preActivation,
        firstStateEntryIndex,
        sumOfChainAndBackwardPreviousState,
        firstSumOfChainAndBackwardPreviousState,
        backwardResult,
        firstStateEntryIndex,
        startEntryIndex,
        exclusiveEndEntryIndex,
        activationFunction);

}

__device__ void backwardStepsInBetween(
    float* preActivation,
    float* hiddenStates,
    float* backwardResult,
    float* chain,
    float* sharedMemory,
    int firstStateEntryIndex,
    float* sumOfChainAndBackwardPreviousState,
    int firstSumOfChainAndBackwardPreviousState,
    float* previousStateWeights,
    float* previousStateWeightAccumulation,
    int firstAccumulatorEntryIndex,
    int startEntryIndex,
    int exclusiveEndEntryIndex,
    int hiddenDimension,
    int activationFunction) {

    // Note that the differentiation w.r.t the previous state is in the second half of shared memory.
    backwardRecurrentActivation(
        preActivation,
        firstStateEntryIndex,
        sumOfChainAndBackwardPreviousState,
        firstSumOfChainAndBackwardPreviousState,
        sharedMemory,
        0,
        startEntryIndex,
        exclusiveEndEntryIndex,
        activationFunction);

    __syncthreads();

    // Differentiate weighted previous state w.r.t previous state and write the result into the second half of shared memory.
    backwardMatrixVectorMultiplicationWrtVector(
        previousStateWeights,
        sharedMemory,
        sharedMemory,
        hiddenDimension,
        startEntryIndex,
        exclusiveEndEntryIndex,
        hiddenDimension);

    int firstPreviousStateEntryIndex = firstStateEntryIndex - hiddenDimension;

    // Differentiate weighted previous state w.r.t weights and add to accumulator
    backwardMatrixVectorMultiplicationWrtMatrix(
        hiddenStates,
        firstPreviousStateEntryIndex,
        sharedMemory,
        0,
        previousStateWeightAccumulation,
        firstAccumulatorEntryIndex,
        startEntryIndex,
        exclusiveEndEntryIndex,
        hiddenDimension);

    copyCooperatively(sharedMemory, 0, backwardResult, firstStateEntryIndex, startEntryIndex, exclusiveEndEntryIndex);

}