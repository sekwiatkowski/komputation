package com.komputation.cuda

import jcuda.Pointer

interface CudaForwardResult {
    val deviceForwardResult: Pointer
    val maximumOutputColumns : Int
}

interface CudaBackwardResult {
    val deviceBackwardResult: Pointer
    val maximumInputColumns : Int
}