extern "C"
__global__ void backwardSquaredLossKernel (int batchSize, int numberInstancePerEntry, float *predictions, float *targets, float *result)
{

    int indexInstance = blockIdx.x;
    int startInstance = indexInstance * numberInstancePerEntry;

    int indexEntryInInstance = threadIdx.x;
    int indexEntryInBatch = startInstance + indexEntryInInstance;

    if(indexInstance < batchSize) {

        result[indexEntryInBatch] = predictions[indexEntryInBatch] - targets[indexEntryInBatch];

    }
    else {

        result[indexEntryInBatch] = 0.0;

    }

}