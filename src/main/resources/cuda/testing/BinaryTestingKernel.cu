__global__ void binaryTestingKernel (
    int batchStart,
    int length,
    float* predictions,
    float* targets,
    int* result) {

    int withinBatch = blockIdx.x;
    int instanceStart = batchStart + withinBatch * length;
    int instanceEnd = instanceStart + length;

    for(int indexEntry = instanceStart; indexEntry < instanceEnd; indexEntry++) {

        float prediction = predictions[indexEntry];
        float target = targets[indexEntry];

        result[indexEntry] = (prediction < 0.5 && target = 0.0) || (prediction >= 0.5 && target = 1.0);

    }

}