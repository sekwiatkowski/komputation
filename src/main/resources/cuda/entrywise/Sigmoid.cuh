__inline__ __device__ float sigmoid (float x) {
    return 1.0 / (1.0 + expf (-x));
}