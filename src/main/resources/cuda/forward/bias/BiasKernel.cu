extern "C"
__global__ void biasKernel (int batchSize, int numberEntriesPerInstance, int numberRows, float* input, float* bias, float* result)
{

    int indexInstance = blockIdx.x;
    int startInstance = indexInstance * numberEntriesPerInstance;
    int indexEntryInInstance = blockIdx.y * blockDim.x + threadIdx.x;
    int indexEntryInBatch = startInstance + indexEntryInInstance;

    if(indexEntryInInstance < numberEntriesPerInstance) {

        if(indexInstance < batchSize) {

            int indexColumn = indexEntryInInstance % numberRows;

            result[indexEntryInBatch] = input[indexEntryInBatch] + bias[indexColumn];

        }
        else {

            result[indexEntryInBatch] = 0.0;

        }

    }

}