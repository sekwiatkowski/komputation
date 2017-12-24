__inline__ __device__ void setToZero(float* destination, int start, int end) {
    for(int index = start; index < end; index++) {
        destination[index] = 0.0;
    }
}