package com.komputation.cuda

import jcuda.Pointer

interface CudaForwardPropagation : CudaForwardState {
    fun forward(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, isTraining: Boolean): Pointer
}

interface CudaBackwardPropagation : CudaBackwardState {
    fun backward(batchSize: Int, chain: Pointer) : Pointer
}