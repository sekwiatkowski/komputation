package com.komputation.cuda

interface CudaForwardDimensions {
    val numberOutputRows : Int
}

interface CudaBackwardDimensions {
    val numberInputRows : Int
}