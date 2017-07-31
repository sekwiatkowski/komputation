extern "C"
__global__ void stochasticGradientDescentKernel (int length, float scalingFactor, float learningRate, float *parameter, float *gradient)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length) {

        parameter[index] -= scalingFactor * learningRate * gradient[index];

    }

}