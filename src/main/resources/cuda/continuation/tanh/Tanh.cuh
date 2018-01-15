#include "../../cuda.h"

__inline__ __device__ float tanh (float x) {
    return (2.0f / (1.0f + expf(-2.0f*x))) - 1.0f;
}

__inline__ __device__ float backwardTanh (float forward, float chain) {
    return chain * (1.0f - powf(forward, 2.0f));
}