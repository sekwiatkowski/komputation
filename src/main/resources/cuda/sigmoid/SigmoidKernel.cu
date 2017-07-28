__device__ float sigmoid (float x)
{

    return 1.0 / (1.0 + expf (-x));

}

extern "C"
__global__ void sigmoidKernel (int batchSize, int numberEntriesPerInstance, float *source, float *destination)
{

    int indexInstance = blockIdx.x;
    int startInstance = indexInstance * numberEntriesPerInstance;
    int indexEntryInInstance = blockIdx.y * blockDim.y + threadIdx.x;
    int indexEntryInBatch = startInstance + indexEntryInInstance;

    if(indexEntryInInstance < numberEntriesPerInstance) {

        if(indexInstance < batchSize) {

            destination[indexEntryInBatch] = sigmoid(source[indexEntryInBatch]);

        }
        else {

            destination[indexEntryInBatch] = 0.0;

        }

    }

}