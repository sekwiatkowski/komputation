__device__ float reduceWarpToSum(float thisValue) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float otherValue = __shfl_down(thisValue, offset, warpSize);

        thisValue += otherValue;
    }

    return thisValue;
}

__device__ void reduceWarpsToSums(float thisValue, int warpId, int laneId, float* shared) {
    float warpSum = reduceWarpToSum(thisValue);

    if(laneId == 0) {
        shared[warpId] = warpSum;
    }

    __syncthreads();
}

__device__ float reduceWarpsToSum(float thisValue, int warpId, int laneId, float* shared) {
    reduceWarpsToSums(thisValue, warpId, laneId, shared);

    int numberWarps = (blockDim.x + warpSize - 1) / warpSize;

    float warpSum = (threadIdx.x < numberWarps) ? shared[threadIdx.x] : 0.0;

    return reduceWarpToSum(warpSum);

}