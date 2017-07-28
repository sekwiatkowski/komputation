__device__ float tanh (float x)
{

    return (2.0 / (1.0 + expf(-2.0*x))) - 1.0;

}

extern "C"
__global__ void tanhKernel (int batchSize, int numberEntriesPerInstance, float *source, float *destination)
{

    int indexInstance = blockIdx.x;
    int startInstance = indexInstance * numberEntriesPerInstance;
    int indexEntryInInstance = blockIdx.y * blockDim.y + threadIdx.x;
    int indexEntryInBatch = startInstance + indexEntryInInstance;

    if(indexEntryInInstance < numberEntriesPerInstance) {

        if(indexInstance < batchSize) {

            destination[indexEntryInBatch] = tanh(source[indexEntryInBatch]);

        }
        else {

            destination[indexEntryInBatch] = 0.0;

        }

    }

}