#include "reduction/Reduction.cuh"

// One block per chain
template <int blockSize>
__global__ void backwardNormalizationKernel (int numberRows, float* chain, float* forward, float* sums, float* results)
{

    int indexRow = threadIdx.x;
    int indexColumn = blockIdx.x;
    int firstIndexOfColumn = indexColumn * numberRows;
    int indexEntry = firstIndexOfColumn + indexRow;

    // Put the products into shared memory
    extern __shared__ float sharedData[];

    if(indexRow < numberRows) {

        sharedData[indexRow] = chain[indexEntry] * -forward[indexEntry];

    }

    __syncthreads();

    // Reduce the products
    reduce<blockSize>(indexRow, sharedData, 0);

    // Compute the result
    results[indexEntry] = (sharedData[0] + chain[indexEntry]) / sums[indexColumn];

}