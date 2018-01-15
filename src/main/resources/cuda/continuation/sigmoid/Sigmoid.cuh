#include "../../cuda.h"

__inline__ __device__ float sigmoid (float x) {
    return 1.0f / (1.0f + expf (-x));
}

__inline__ __device__ float backwardSigmoid (float forward, float chain) {
    return forward * (1.0f - forward) * chain;
}