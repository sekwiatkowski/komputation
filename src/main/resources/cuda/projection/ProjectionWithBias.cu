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
__global__ void projectionWithBiasKernel(double * input, double * weights, double * bias, double * result, int numberRows, int numberColumns)
{

    int indexRow = threadIdx.x + blockIdx.x * blockDim.x;

    // All threads in a block have access to shared memory.
    extern __shared__ double sharedData[];

    double resultEntry = 0.0;

    // Columns are processed in blocks.
    // numberColumns = 1, blockDim.x = 16: (numberColumns + blockDim.x - 1) / BLOCK__SIZE = (1 + 16 - 1) / 16 = 1
    // numberColumns = 2, blockDim.x = 16: (2 + 16 - 1) / 16 = 17 / 16 = 1
    // numberColumns = 15, blockDim.x = 16: (15 + 16 - 1) / 16 = 30 / 16 = 1
    // numberColumns = 16, blockDim.x = 16: (16 + 16 - 1) / 16 = 31 / 16 = 1
    // numberColumns = 17, blockDim.x = 16: (17 + 16 - 1) / 16 = 32 / 16 = 2
    // numberColumns = 32, blockDim.x = 16: (32 + 16 - 1) / 16 = 47 / 16 = 2
    // numberColumns = 33, blockDim.x = 16: (33 + 16 - 1) / 16 = 48 / 16 = 3
    // 1-16: 1, 17-32: 2, 33-48: 3

    // Adding blockDim.x ensures a result of at least one ((1+16-1)/16=16/16=1)
    // Otherwise: 1-16: 0, 17-32: 1, 33-48: 2

    // Subtracting one ensures that all blocks are full.
    // Otherwise: 1-15: 1 (missing #16), 16-31: 2 (missing #32), 32-47: 3 (missing #38)

    // #pragma unroll is a compiler optimization
    #pragma unroll
    // While the number of blocks is a function of the number of rows, the number of steps in this loop is determined by the number of columns.
    for (int outerIndex = 0; outerIndex < (numberColumns + blockDim.x - 1) / blockDim.x; outerIndex++)
    {

        // Put the data into shared memory.
        // This uses the thread ID, not the global ID.
        // Suppose there are 2 threads per block (blockDim.x = 2).
        // Block    Thread    Thread ID + block ID * blockDim.x
        //                                       (column index)
        //     0         0                        0 + 0 * 2 = 0
        //     0         1                        1 + 0 * 2 = 1
        //     1         0                        0 + 1 * 2 = 2
        //     1         1                        1 + 1 * 2 = 3
        // The first block processes the first two columns (0 and 1), the second block processes columns 2 and 3, etc.
        int indexColumn = threadIdx.x + outerIndex * blockDim.x;

        // The first thread in the first block puts odd vector entries (1, 3, 5, ...) into the shared memory.
        // The second thread in the first block puts even vector entries (0, 2, 4, ...) into the shared memory.
        if (indexColumn < numberColumns) {

            sharedData[threadIdx.x] = input[indexColumn];

        }

        // Guarantee that every thread in the block has been completed
        __syncthreads();

        // Another compiler optimization
        #pragma unroll
        // From 0 to blockDim.x-1
        for (int innerIndex = 0; innerIndex < blockDim.x; innerIndex++) {

            // Note that this is in column-major order.
            // Id    innerIndex    outerIndex    blockDim.x * outerIndex    blockDim.x * outerIndex + innerIndex    blockDim.x * outerIndex + innerIndex * numberRows
            //                                   (1st column in current block)                           (column)
            //  0             0             0                          0                                       0                                                    0
            //  1             1             0                          0                                       1                                                    2
            //  2             0             1                          2                                       2                                                    4
            //  3             1             1                          2                                       3                                                    6

            // The blockDim.x is 2.
            // The number of blocks is given by: (numberRows + blockDim.x - 1) / blockDim.x
            // The total number of threads is given by: number of blocks * blockDim.x
            // 1x2 matrix: number of blocks = (1 + 2 - 1) / 2 = 2 / 2 = 1 = 1, total number of threads = 1 * 2 = 2
            // 2x2 matrix: number of blocks = (2 + 2 - 1) / 2 = 3 / 2 = 1.5 = 1, total number of threads = 1 * 2 = 2
            // 3x2 matrix: number of blocks = (3 + 2 - 1) / 2 = 4 / 2 = 2, total number of threads = 2 * 2 = 4
            // 4x4 matrix: number of blocks = (4 + 2 - 1) / 2 = 5 / 2 = 2.5 = 2, total number of threads = 2 * 2 = 4
            // Total number of threads = ceil(number of rows / blockDim.x)
            // Note that the total number of threads is independent of the number of columns.

            // blockDim.x * outerIndex + innerIndex is the column index
            // indexRow + (blockDim.x * outerIndex + innerIndex) * numberRows is the entry index.
            resultEntry += weights[indexRow + (blockDim.x * outerIndex + innerIndex) * numberRows] * sharedData[innerIndex];

        }

        // Guarantee that every thread in the block has been completed
        __syncthreads();

    }

    // Set the result entries in the threads with the first numberRows global IDs
    if (indexRow < numberRows) {

        result[indexRow] = resultEntry + bias[indexRow];

    }

}