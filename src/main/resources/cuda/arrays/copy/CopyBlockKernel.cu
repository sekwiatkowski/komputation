#include "../../cuda.h"
#include "../../symbols/NaN.cuh"

__global__ void copyBlockKernel (
    int batchSize,
    int numberIterations,
    float* source,
    int sourceMaximumEntries,
    int sourceNumberRows,
    int firstRowInSource,
    float *destination,
    int destinationMaximumEntries,
    int destinationNumberRows,
    int firstRowInDestination) {

    int instanceIndex = blockIdx.x;
    int columnIndex = blockIdx.y;

    int firstDestinationEntryIndexInCurrentColumn = instanceIndex * destinationMaximumEntries + columnIndex * destinationNumberRows + firstRowInDestination;
    int firstDestinationEntryIndex = firstDestinationEntryIndexInCurrentColumn + threadIdx.x * numberIterations;

    int destinationEntryIndex = firstDestinationEntryIndex;

    int exclusiveLastDestinationEntryIndex = min(firstDestinationEntryIndex + numberIterations, firstDestinationEntryIndexInCurrentColumn + sourceNumberRows);

    if(instanceIndex < batchSize) {
        int firstSourceEntryIndex = instanceIndex * sourceMaximumEntries + columnIndex * sourceNumberRows + firstRowInSource + threadIdx.x * numberIterations;
        int sourceEntryIndex = firstSourceEntryIndex;

        while(destinationEntryIndex < exclusiveLastDestinationEntryIndex) {
            destination[destinationEntryIndex] = source[sourceEntryIndex];

            sourceEntryIndex++;
            destinationEntryIndex++;
        }

    }
    else {
        setToNaN(destination, destinationEntryIndex, exclusiveLastDestinationEntryIndex);
    }

}