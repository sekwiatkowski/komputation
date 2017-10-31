__global__ void multiClassTestingKernel (
    int batchStart,
    int numberRows,
    int numberColumns,
    int numberEntriesPerInstance,
    float* predictions,
    float* targets,
    int* result) {

    int indexInstance = blockIdx.x;
    int instanceStart = indexInstance * numberEntriesPerInstance;

    for(int indexColumn = 0; indexColumn < numberColumns; indexColumn++) {

        int instanceColumnStart = instanceStart + indexColumn * numberRows;
        int instanceColumnEnd = instanceColumnStart + numberRows;

        float maximumPrediction = 0;
        int maximumPredictionIndex = -1;

        float maximumTarget = 0;
        int maximumTargetIndex = -1;

        for(int indexEntry = instanceColumnStart; indexEntry < instanceColumnEnd; indexEntry++) {

            float prediction = predictions[indexEntry];
            float target = targets[indexEntry];

            if(prediction > maximumPrediction) {

                maximumPrediction = prediction;
                maximumPredictionIndex = indexEntry;

            }

            if(target > maximumTarget) {

                maximumTarget = target;
                maximumTargetIndex = indexEntry;

            }

        }

        if(maximumPredictionIndex != maximumTargetIndex) {

            result[batchStart + indexInstance] = 0;

            return;

        }

    }

    result[batchStart + indexInstance] = 1;

}