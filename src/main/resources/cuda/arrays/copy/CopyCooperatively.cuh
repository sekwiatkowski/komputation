#include "../../cuda.h"

__device__ void copyCooperatively(float* source, int firstSourceEntry, float* destination, int firstDestinationEntry, int startEntryIndex, int exclusiveEndEntryIndex) {
    for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
        destination[firstDestinationEntry + entryIndex] = source[firstSourceEntry + entryIndex];
    }
}