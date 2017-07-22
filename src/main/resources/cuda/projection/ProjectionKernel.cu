#include "projection/Projection.cuh"

// The number of blocks is given by:
// (numberRows + blockDim.x - 1) / blockDim.x
// It's a function of the number of rows. It's not a function of the number of columns.
// Ex.: blockDim.x = 16
// numberRows = 1: 1 + 16 - 1 / 16 = 16 / 16 = 1
// numberRows = 2: 2 + 16 - 1 / 16 = 17 / 16 = 1
// numberRows = 17: 17 + 16 - 1 / 16 = 32 / 16 = 2

// Each thread is responsible for a row.
// The first thread in the first block processes the first row.
// The second thread in the first block processes the second row.
// The first thread in the second block processes the third row.
// The second thread in the second block processes the fourth row.

// Number of blocks = (numberRows + blockDim.x - 1) / blockDim.x
// Number of threads = blockDim.x
// Shared memory size = blockDim.x * DOUBLE_SIZE
extern "C"
__global__ void projectionKernel(float * input, float * weights, int numberRows, int numberColumns, float * result)
{

    int indexRow = threadIdx.x + blockIdx.x * blockDim.x;

    // All threads in a block have access to shared memory.
    extern __shared__ float sharedData[];

    float resultEntry = project(input, weights, numberRows, numberColumns, indexRow, sharedData);

    // Set the result entries in the threads with the first numberRows global IDs
    if (indexRow < numberRows) {

        result[indexRow] = resultEntry;

    }

}