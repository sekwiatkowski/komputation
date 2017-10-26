__inline__ __device__ float warpReduceToProduct(float thisValue) {

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {

        float otherValue = __shfl_down(thisValue, offset, warpSize);

        thisValue *= otherValue;

    }

    return thisValue;

}

__device__ void reduceToProduct(float thisValue, int warpId, int laneId, float* shared) {

    float warpProduct = warpReduceToProduct(thisValue);

    if(laneId == 0) {

        shared[warpId] = warpProduct;

    }

    __syncthreads();

    thisValue = (threadIdx.x < blockDim.x / warpSize) ? shared[laneId] : 0.0;

    if (warpId == 0) {

        warpReduceToProduct(thisValue);

    }

}