template <int blockSize>
__device__ void reduceWarp(volatile double* sharedData, int indexSharedData) {

    if(blockSize >= 64) sharedData[indexSharedData] += sharedData[indexSharedData + 32];
    if(blockSize >= 32) sharedData[indexSharedData] += sharedData[indexSharedData + 16];
    if(blockSize >= 16) sharedData[indexSharedData] += sharedData[indexSharedData + 8];
    if(blockSize >= 8) sharedData[indexSharedData] += sharedData[indexSharedData + 4];
    if(blockSize >= 4) sharedData[indexSharedData] += sharedData[indexSharedData + 2];
    if(blockSize >= 2) sharedData[indexSharedData] += sharedData[indexSharedData + 1];

}

// This function works based on the assumptions that the number of threads per block is set to a power of 2.
template <int blockSize>
__device__ void reduce(int threadId, double* sharedData, int offsetSharedData) {

    int indexSharedData = offsetSharedData + threadId;

    if(blockSize >= 512) {

        sharedData[indexSharedData] += sharedData[indexSharedData + 256];

        __syncthreads();

    }

    if(blockSize >= 256) {

        sharedData[indexSharedData] += sharedData[indexSharedData + 128];

        __syncthreads();

    }

    if(blockSize >= 128) {

        sharedData[indexSharedData] += sharedData[indexSharedData + 64];

        __syncthreads();

    }

    // All warps in a block except for the first one can now be ignored.
    // The condition threadId < 32 returns true only for the first warp. A local variable name would be "isFirstWrap".
    if (threadId < 32) {

        reduceWarp<blockSize>(sharedData, indexSharedData);

    }

}