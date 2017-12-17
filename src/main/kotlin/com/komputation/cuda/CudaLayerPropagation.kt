package com.komputation.cuda

import jcuda.Pointer

interface CudaForwardPropagation : CudaForwardState {
    fun forward(batchSize: Int, deviceInput: Pointer, isTraining: Boolean): Pointer
}

interface CudaVariableLengthForwardPropagation {
    fun forward(batchSize: Int, deviceLengths: Pointer, deviceInput: Pointer, isTraining: Boolean): Pointer
}

interface CudaBackwardPropagation : CudaBackwardState {
    fun backward(batchSize: Int, chain: Pointer) : Pointer
}