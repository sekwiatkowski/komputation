extern "C"
__global__ void exponentiationKernel (int batchSize, int numberEntriesPerInstance, float *source, float *destination)
{

    int indexInstance = blockIdx.x;
    int startInstance = indexInstance * numberEntriesPerInstance;
    int indexEntryInInstance = blockIdx.y * blockDim.y + threadIdx.x;
    int indexEntryInBatch = startInstance + indexEntryInInstance;

    if(indexEntryInInstance < numberEntriesPerInstance) {

        if(indexInstance < batchSize) {

            destination[indexEntryInBatch] = expf(source[indexEntryInBatch]);

        }
        else {

            destination[indexEntryInBatch] = 0.0;

        }

    }

}