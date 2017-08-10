extern "C"
__global__ void stochasticGradientDescentKernel (
    int numberIterations,
    float learningRate,
    int* parameterIndices,
    int parameterSize,
    float* parameters,
    float scalingFactor,
    float* gradient)
{

    int startEntry = (blockIdx.y * blockDim.x * numberIterations) + threadIdx.x * numberIterations;

    if(startEntry < parameterSize) {

        int indexGradient = blockIdx.x;
        int indexParameter = parameterIndices[indexGradient];

        int startParameter = indexParameter * parameterSize + startEntry;
        int startGradient = indexGradient * parameterSize + startEntry;

        for(int i = 0; i < numberIterations; i++) {

            parameters[startParameter + i] -= scalingFactor * learningRate * gradient[startGradient + i];

        }

    }

}