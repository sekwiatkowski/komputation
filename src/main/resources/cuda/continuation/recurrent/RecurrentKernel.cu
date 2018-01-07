#include "constants/Activation.cuh"
#include "constants/ResultExtraction.cuh"
#include "entrywise/Relu.cuh"
#include "entrywise/Sigmoid.cuh"
#include "entrywise/Tanh.cuh"
/*
    number of blocks in x dimension = number of instances
    number of threads = (hidden dimension +  maximum number of threads per block - 1) / maximum number of threads per block
    shared memory size = state size

    This kernel assumes that the input has already been projected.
*/
__global__ void recurrentKernel (
    int activationFunction,
    int maximumEntriesPerInstance,
    int hiddenDimension,
    int numberIterations,
    float* projectedInput,
    int *lengths,
    int resultExtraction,
    float* result) {

    int instanceIndex = blockIdx.x;

    // First step in each instance
    int firstInstanceEntryIndex = instanceIndex * maximumEntriesPerInstance;

    int startEntryIndex = threadIdx.x * numberIterations;
    // Do not go past the hidden dimension
    int exclusiveEndEntryIndex = min(startEntryIndex + numberIterations, hiddenDimension);

    extern __shared__ float sharedData[];

    // Copy the first step of the projected input into shared memory
    for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
        sharedData[entryIndex] = projectedInput[firstInstanceEntryIndex + entryIndex];
    }

    // Apply the entrywise activation function
    switch(activationFunction) {
        case RELU:
            for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
                sharedData[entryIndex] = relu(sharedData[entryIndex]);
            }
            break;
        case SIGMOID:
            for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
                sharedData[entryIndex] = sigmoid(sharedData[entryIndex]);
            }
            break;
        case TANH:
            for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
                sharedData[entryIndex] = tanh(sharedData[entryIndex]);
            }
            break;
    }

    if(resultExtraction == ALL_STEPS) {
        for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
            result[result] = sharedData[entryIndex];
        }
    }

    int length = lengths[instanceIndex];

    for(int step = 1; step < length; step++) {

    }

}