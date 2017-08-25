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

        for(int indexParameter = startParameter, indexGradient = startGradient; indexParameter < startParameter + numberIterations; indexParameter++, indexGradient++) {

            parameters[indexParameter] -= scalingFactor * learningRate * gradient[indexGradient];

        }

    }

}