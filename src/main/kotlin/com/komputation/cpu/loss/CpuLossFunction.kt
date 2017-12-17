package com.komputation.cpu.loss

import com.komputation.cpu.layers.CpuBackwardResult


interface CpuLossFunction : CpuBackwardResult {

    fun forward(numberInputColumns : Int, predictions: FloatArray, targets : FloatArray): Float

    fun backward(predictions: FloatArray, targets : FloatArray) : FloatArray

}