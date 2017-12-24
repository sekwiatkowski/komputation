package com.komputation.cuda

interface CudaForwardDimensions {
    val numberOutputRows : Int
    val maximumOutputColumns : Int
}

interface CudaBackwardDimensions {
    val numberInputRows : Int
    val maximumInputColumns : Int
}