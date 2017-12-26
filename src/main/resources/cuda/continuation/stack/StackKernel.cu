__global__ void stackKernel (
    int sourceMaximumEntries,
    int sourceNumberRows,
    int destinationMaximumEntries,
    int destinationNumberRows,
    int firstRowInDestination,
    float* source,
    float *destination,
    int numberIterations) {

    int instanceIndex = blockIdx.x;
    int columnIndex = blockIdx.y;

    int firstSourceEntryIndex = instanceIndex * sourceMaximumEntries + columnIndex * sourceNumberRows + threadIdx.x * numberIterations;

    if(firstSourceEntryIndex < sourceMaximumEntries) {

        int firstDestinationEntryIndexWithinBatch = instanceIndex * destinationMaximumEntries;
        int firstDestinationEntryIndexInCurrentColumnWithinBatch = firstDestinationEntryIndexWithinBatch + columnIndex * destinationNumberRows + firstRowInDestination;
        int firstDestinationEntryIndex = firstDestinationEntryIndexInCurrentColumnWithinBatch + threadIdx.x * numberIterations;

        int sourceEntryIndex = firstSourceEntryIndex;
        int destinationEntryIndex = firstDestinationEntryIndex;

        int lastDestinationEntryIndexInCurrentColumn = firstDestinationEntryIndexInCurrentColumnWithinBatch + sourceNumberRows;

        while(destinationEntryIndex < min(firstDestinationEntryIndex + numberIterations, lastDestinationEntryIndexInCurrentColumn)) {

            destination[destinationEntryIndex] = source[sourceEntryIndex];

            sourceEntryIndex++;
            destinationEntryIndex++;
        }

    }

}