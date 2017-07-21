// -1/target probability if target = 1.0, 0.0 otherwise
__global__ void backwardLogisticLossKernel (double *predictions, double *targets, double *result)
{

    int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    result[globalId] = targets[globalId] * -(1/predictions[globalId]);

}