#include "reduction/Reduction.cuh"

/*

    d a/(a+b) / d a = b/(a+b)^2
    d b/(a+b) / d a = -b/(a+b)^2

    d a/(a+b+c) / d a = (b+c)/(a+b+c)^2
    d b/(a+b+c) / d a = -b/(a+b+c)^2
    d c/(a+b+c) / d a = -c/(a+b+c)^2

*/
template <int blockSize>
__global__ void backwardNormalizationKernel (int numberCategories, double* inputs, double* sums, double* chain, double* results)
{

    int indexRow = threadIdx.x;
    int indexColumn = blockIdx.x;
    int numberRows = blockDim.x;
    int firstIndexOfColumn = indexColumn * numberRows;
    int indexEntry = firstIndexOfColumn + indexRow;

    extern __shared__ double sharedData[];

    if(indexRow == 0) {

        // The first entry contains the squared sum for the current column.
        sharedData[0] = sums[indexColumn] * sums[indexColumn];

    }

    // The rest of the shared memory contains the multiplication of the chain entries with the input entries.
    // Recall that the block size is a power of 2.
    // In many cases the number of threads will thus exceed the number of categories.
    double thisMultiplication = -1;

    if(indexRow < numberCategories) {

        thisMultiplication = chain[indexEntry] * inputs[indexEntry];

        sharedData[1+indexRow] = thisMultiplication;

    }

    __syncthreads();

    reduce<blockSize>(indexRow, sharedData, 1);

    /*
        sharedData[0] still contains the squared sum.
        sharedData[1] holds the sum over the multiplications.
    */

    double sameEntry = chain[indexEntry] * (sums[indexColumn] - inputs[indexEntry]);
    double otherEntries = -sharedData[1] + thisMultiplication;

    results[indexEntry] = (sameEntry + otherEntries) / sharedData[0];

}