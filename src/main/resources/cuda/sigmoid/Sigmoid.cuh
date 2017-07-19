__device__ double sigmoid (double x)
{

    return 1.0 / (1.0 + exp (-x));

}