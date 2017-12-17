package com.komputation.cpu.layers

interface CpuForwardPropagation : CpuForwardState {
    fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean) : FloatArray
}

interface CpuBackwardPropagation : CpuBackwardState {
    fun backward(withinBatch : Int, chain : FloatArray) : FloatArray
}