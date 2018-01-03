package com.komputation.cuda

import jcuda.Pointer

interface CudaForwardResult {
    val deviceForwardResult: Pointer
    val deviceForwardLengths: Pointer
    val largestNumberOutputColumnsInCurrentBatch: Int
}

interface CudaBackwardResult {
    val deviceBackwardResult: Pointer
    val largestNumberInputColumnsInCurrentBatch: Int
}