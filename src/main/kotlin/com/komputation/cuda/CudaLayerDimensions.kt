package com.komputation.cuda

interface CudaForwardDimensions {
    val numberOutputRows : Int
    val maximumOutputColumns : Int
    val maximumOutputEntries : Int
}

interface CudaBackwardDimensions {
    val numberInputRows : Int
    val maximumInputColumns : Int
    val maximumInputEntries : Int
}