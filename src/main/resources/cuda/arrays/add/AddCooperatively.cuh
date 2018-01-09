__device__ void addCooperatively(float* source, int firstSourceEntry, float* destination, int firstDestinationEntry, int startEntryIndex, int exclusiveEndEntryIndex) {
    for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
        destination[firstDestinationEntry + entryIndex] += source[firstSourceEntry + entryIndex];
    }
}

__device__ void addCooperatively(float* a, int firstAEntry, float* b, int firstBEntry, float* result, int firstResultEntry, int startEntryIndex, int exclusiveEndEntryIndex) {
    for(int entryIndex = startEntryIndex; entryIndex < exclusiveEndEntryIndex; entryIndex++) {
         result[firstResultEntry + entryIndex] += a[firstAEntry + entryIndex] + b[firstBEntry + entryIndex];
    }
}