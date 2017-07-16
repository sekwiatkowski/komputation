extern "C"
__global__ void stochasticGradientDescentKernel (int length, double *parameter, double scalingFactor, double learningRate, double *gradient)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        parameter[index] = parameter[index] - scalingFactor * learningRate * gradient[index];
        gradient[index] = 0.0;

    }

    __syncthreads();

}