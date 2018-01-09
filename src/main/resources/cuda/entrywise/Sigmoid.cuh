__inline__ __device__ float sigmoid (float x) {
    return 1.0 / (1.0 + expf (-x));
}

__inline__ __device__ float backwardSigmoid (float forward, float chain) {
    return forward * (1.0 - forward) * chain;
}