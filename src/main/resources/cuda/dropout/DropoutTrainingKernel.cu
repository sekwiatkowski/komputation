__device__  int xorShift(int seed) {

    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;

    return seed;

}

/*
    dropout probability is 1 - keep probability and should be less than 1.
    seed + 2147483648.0: [0, 2^32/2 + 2^32/2-1 = 4294967295]
    (seed + 2147483648.0) / 4294967295.0: [0 to 1]
    (seed + 2147483648.0) / 4294967295.0 - dropout probability): (0 to 1]
    ceilf(seed + 2147483648.0) / 4294967295.0 - dropout probability): or or 1
*/

__device__ float generateMask(float seed, float dropoutProbability) {

    return ceilf((seed + 2147483648.0) / 4294967295.0 - dropoutProbability);

}

extern "C"
__global__ void dropoutTrainingKernel (int numberEntries, float dropoutProbability, float* input, int* seeds, float* masks, float* result)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < numberEntries) {

        int newSeed = xorShift(seeds[index]);
        seeds[index] = newSeed;

        float mask = generateMask((float)newSeed, dropoutProbability);
        masks[index] = mask;

        result[index] = mask * input[index];

    }

}