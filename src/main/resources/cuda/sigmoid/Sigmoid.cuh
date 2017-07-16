#pragma once

__device__ inline double sigmoid (double x)
{

    return 1.0 / (1.0 + exp (-x));

}

__device__ inline double backwardSigmoid (double forward, double chain)
{

    return forward * (1.0 - forward) * chain;

}