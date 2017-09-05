__device__ int tryToInsert(int* mutex, int id, int value) {

    int result = atomicCAS((int*) (mutex + id), -1, value);

    if (result == -1 || result == value)
        return 1;
    else
        return 0;

}

__global__ void hashKernel(
    int* indices,
    int hashTableSize,
    int* hashTable,
    int* counts,
    int* mapping) {

    extern __shared__ int instanceCounts[];

    int indexInstance = blockIdx.x;
    int indexColumn = threadIdx.x;
    int maximumNumberColumns = blockDim.x;
    int indexColumnWithinBatch = indexInstance * maximumNumberColumns + indexColumn;

    int stepInstanceCount = hashTableSize / maximumNumberColumns;
    int startInstanceCount = threadIdx.x * stepInstanceCount;
    int endInstanceCount = startInstanceCount + stepInstanceCount;
    for(int indexInstanceCount = startInstanceCount; indexInstanceCount < endInstanceCount; indexInstanceCount++) {

        instanceCounts[indexInstanceCount] = 0;

    }

    mapping[indexColumnWithinBatch] = -1;

    __syncthreads();

    int parameterIndex = indices[indexColumnWithinBatch];

    if(parameterIndex != -1) {

        unsigned candidate = parameterIndex % hashTableSize;

        while(true) {

            int insertionResult = tryToInsert(hashTable, candidate, parameterIndex);

            if(insertionResult == 1) {

                atomicExch(&instanceCounts[candidate], 1);
                mapping[indexColumnWithinBatch] = candidate;

                break;

            }
            else {

                candidate = (candidate + 1) % hashTableSize;

            }

        }

    }

    __syncthreads();

    for(int indexInstanceCount = startInstanceCount; indexInstanceCount < endInstanceCount; indexInstanceCount++) {

        atomicAdd(&counts[indexInstanceCount], instanceCounts[indexInstanceCount]);

    }

}