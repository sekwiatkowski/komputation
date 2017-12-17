package com.komputation.cpu.layers

interface CpuForwardPropagation : CpuVariableLengthForwardState {
    val numberOutputRows: Int
    fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean) : FloatArray
}

interface CpuBackwardPropagation : CpuVariableLengthBackwardState {
    val numberInputRows: Int
    fun backward(withinBatch : Int, chain : FloatArray) : FloatArray
}

interface CpuForwardLayer : CpuForwardPropagation, CpuBackwardPropagation