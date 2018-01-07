__inline__ __device__ float tanh (float x) {
    return (2.0 / (1.0 + expf(-2.0*x))) - 1.0;
}