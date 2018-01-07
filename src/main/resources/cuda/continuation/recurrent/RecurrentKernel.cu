__global__ void recurrentKernel (
    float* initialState,
    float* input,
    float* result,
    int *lengths,
    int hiddenDimension,
    int numberIterations) {

    int instanceIndex = blockIdx.x;
    int length = lengths[instanceIndex];

    int startEntryIndex = threadIdx.x * numberIterations;
    int exclusiveEndEntryIndex = min(startEntryIndex + numberIterations, hiddenDimension);

    extern __shared__ float sharedData[];

    for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
        sharedData[entryIndex] = input[entryIndex];
    }

    __syncThreads();


    for(int step = 1; step < length; step++) {



        // __syncThreads();

    }

}