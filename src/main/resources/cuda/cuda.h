#ifdef __JETBRAINS_IDE__
    #define __device__
    #define __global__
    #define __inline__
    #define __shared__
    #define __constant__

    typedef unsigned int uint;
    typedef struct uint3{
        uint x;
        uint y;
        uint z;
    } uint3;
    typedef uint3 dim3;

    extern dim3 gridDim;
    extern uint3 blockIdx;

    extern dim3 blockDim;
    extern uint3 threadIdx;

    extern int warpSize;

    inline void __syncthreads() {}

    int __shfl_down(float var, unsigned int offset, int width=warpSize);

    float __int_as_float(int x);

    float min(float a, float b);
    int min(int a, int b);

    float fmaxf(float a, float b);

    bool isnan(float x);

    float ceilf(float x);
    float expf(float x);
    float powf(float base, float exponent);
    float log2f(float x);
    float logf(float x);
    float sqrtf(float x);

    float nanf(const char *x);
#endif