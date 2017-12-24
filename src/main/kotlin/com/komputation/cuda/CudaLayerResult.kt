package com.komputation.cuda

import jcuda.Pointer

interface CudaForwardResult {
    val deviceForwardResult: Pointer
    val deviceForwardLengths: Pointer
    val batchMaximumOutputColumns : Int
}

interface CudaBackwardResult {
    val deviceBackwardResult: Pointer
    val batchMaximumInputColumns : Int
}