package com.komputation.cpu.layers

interface CpuForwardLayer : CpuVariableLengthForwardState, CpuVariableLengthBackwardState {

    fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean) : FloatArray

    fun backward(withinBatch : Int, chain : FloatArray) : FloatArray

}