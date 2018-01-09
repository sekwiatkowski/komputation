#include "recurrent/RecurrentActivation.cuh"
#include "entrywise/Relu.cuh"
#include "entrywise/Sigmoid.cuh"
#include "entrywise/Tanh.cuh"
#include "arrays/copy/CopyCooperatively.cuh"
#include "arrays/add/AddCooperatively.cuh"

__device__ void forwardRecurrentActivation(float* input, int startInput, float* result, int startEntryIndex, int exclusiveEndEntryIndex, int activationFunction) {
    switch(activationFunction) {
        case IDENTITY:
            for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
                result[entryIndex] = input[startInput + entryIndex];
            }
        case RELU:
            for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
                result[entryIndex] = relu(input[startInput + entryIndex]);
            }
            break;
        case SIGMOID:
            for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
                result[entryIndex] = sigmoid(input[startInput + entryIndex]);
            }
            break;
        case TANH:
            for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
                result[entryIndex] = tanh(input[startInput + entryIndex]);
            }
            break;
    }
}

/*
    This function is based on three assumptions:
    1) The size of the "vector" array is twice the number of its entries.
    2) The entries are copy into the second half of the array.
    3) The results are written into the first half.
*/
// The vector is in shared memory.
__device__ void matrixVectorMultiplication(float* matrix, float* vector, int startEntryIndex, int exclusiveEndEntryIndex, int dimension) {

    // Duplicate the vector
    copyCooperatively(vector, 0, vector, dimension, startEntryIndex, exclusiveEndEntryIndex);

    __syncthreads();

    // Write the result in the first half of the shared memory, using the entries from the second half.
    for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {

        int matrixRow = entryIndex;

        float entryResult = 0.0;

        for(int matrixColumn = 0; matrixColumn < dimension; matrixColumn++) {

            float matrixEntry = matrix[matrixColumn * dimension + matrixRow];
            float vectorEntry = vector[dimension + matrixColumn];

            entryResult += matrixEntry * vectorEntry;
        }

        vector[entryIndex] = entryResult;
    }

}

__device__ void forwardFirstStep(
    float* projectedInput,
    float* preActivation,
    int firstStateEntryIndex,
    float* result,
    int startEntryIndex,
    int exclusiveEndEntryIndex,
    int activationFunction) {

    copyCooperatively(projectedInput, firstStateEntryIndex, result, 0, startEntryIndex, exclusiveEndEntryIndex);

    copyCooperatively(result, 0, preActivation, firstStateEntryIndex, startEntryIndex, exclusiveEndEntryIndex);

    forwardRecurrentActivation(result, 0, result, startEntryIndex, exclusiveEndEntryIndex, activationFunction);
}

__device__ void forwardOtherStep(
    float* projectedInput,
    float* preActivation,
    float* previousState,
    float* weights,
    int firstStateEntryIndex,
    int startEntryIndex,
    int exclusiveEndEntryIndex,
    int activationFunction,
    int dimension) {

    /*
        The previous hidden state is stored in the shared memory.

                     h1
                     h2
                     h3
        w11 w12 w13
        w21 w22 w23
        w31 w32 w33
    */

    // Weigh the previous hidden state
    matrixVectorMultiplication(weights, previousState, startEntryIndex, exclusiveEndEntryIndex, dimension);

    // At this point, the weighted previous state is in (the first half) of shared memory.

    // Cooperatively add the projected input at the current step to the weighted previous state
    addCooperatively(projectedInput, firstStateEntryIndex, previousState, 0, startEntryIndex, exclusiveEndEntryIndex);

    copyCooperatively(previousState, 0, preActivation, firstStateEntryIndex, startEntryIndex, exclusiveEndEntryIndex);

    // Activate the sum
    forwardRecurrentActivation(previousState, 0, previousState, startEntryIndex, exclusiveEndEntryIndex, activationFunction);

}