__device__ void setToZero(float* destination, int start, int numberIterations) {

    for(int indexEntry = start; indexEntry < start + numberIterations; indexEntry++) {

        destination[indexEntry] = 0.0;

    }

}