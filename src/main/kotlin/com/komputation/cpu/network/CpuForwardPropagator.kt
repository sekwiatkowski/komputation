package com.komputation.cpu.network

import com.komputation.cpu.layers.CpuEntryPoint
import com.komputation.cpu.layers.CpuContinuation
import com.komputation.cpu.layers.CpuForwardResult
import com.komputation.matrix.Matrix

class CpuForwardPropagator(
    private val entryPoint: CpuEntryPoint,
    private val layers : Array<CpuContinuation>) {

    fun forward(withinBatch : Int, input : Matrix, isTraining : Boolean) : CpuForwardResult {
        this.entryPoint.forward(input)

        var currentResult : CpuForwardResult = this.entryPoint

        for (layer in this.layers) {
            layer.forward(withinBatch, currentResult.numberOutputColumns, currentResult.forwardResult, isTraining)
            currentResult = layer
        }

        return currentResult
    }

}