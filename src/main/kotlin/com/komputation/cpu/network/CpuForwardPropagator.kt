package com.komputation.cpu.network

import com.komputation.cpu.layers.CpuEntryPoint
import com.komputation.cpu.layers.CpuForwardLayer
import com.komputation.cpu.layers.CpuForwardResult
import com.komputation.matrix.Matrix

class CpuForwardPropagator(
    private val entryPoint: CpuEntryPoint,
    private val layers : Array<CpuForwardLayer>) {

    fun forward(withinBatch : Int, input : Matrix, isTraining : Boolean) : CpuForwardResult {
        this.entryPoint.forward(input)

        var previousLayerState : CpuForwardResult = this.entryPoint

        for (layer in this.layers) {
            layer.forward(withinBatch, previousLayerState.numberOutputColumns, previousLayerState.forwardResult, isTraining)
            previousLayerState = layer
        }

        return previousLayerState
    }

}