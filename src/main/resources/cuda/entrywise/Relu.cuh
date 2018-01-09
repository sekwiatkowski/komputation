__inline__ __device__ float relu (float x)
{
    return fmaxf(x, 0.0);
}

__inline__ __device__ float backwardRelu (float forward, float chain) {
    if(forward > 0.0) {
        return chain;
    }
    else {
        return 0.0;
    }
}
