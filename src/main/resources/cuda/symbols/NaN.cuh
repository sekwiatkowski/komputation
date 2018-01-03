__inline__ __device__ void setToNaN(float* destination, int start, int end) {
    for(int index = start; index < end; index++) {
        destination[index] = nanf("NaN");
    }
}