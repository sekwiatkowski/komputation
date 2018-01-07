package com.komputation.cuda

import jcuda.Pointer

interface CudaForwardResult {
    val deviceForwardResult: Pointer
    val deviceForwardLengths: Pointer
}

interface CudaBackwardResult {
    val deviceBackwardResult: Pointer
}