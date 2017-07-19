template <int blockSize>
__device__ void reduceWarp(volatile double* sharedData, int threadId) {

    if(blockSize >= 64) sharedData[threadId] += sharedData[threadId + 32];
    if(blockSize >= 32) sharedData[threadId] += sharedData[threadId + 16];
    if(blockSize >= 16) sharedData[threadId] += sharedData[threadId + 8];
    if(blockSize >= 8) sharedData[threadId] += sharedData[threadId + 4];
    if(blockSize >= 4) sharedData[threadId] += sharedData[threadId + 2];
    if(blockSize >= 2) sharedData[threadId] += sharedData[threadId + 1];

}