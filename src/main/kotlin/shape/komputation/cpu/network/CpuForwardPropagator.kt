package shape.komputation.cpu.network

import shape.komputation.cpu.layers.CpuEntryPoint
import shape.komputation.cpu.layers.CpuForwardLayer
import shape.komputation.cpu.layers.CpuForwardState
import shape.komputation.matrix.Matrix

class CpuForwardPropagator(
    private val entryPoint: CpuEntryPoint,
    private val layers : Array<CpuForwardLayer>) {

    fun forward(withinBatch : Int, input : Matrix, isTraining : Boolean) : FloatArray {

        this.entryPoint.forward(input)

        var previousLayerState : CpuForwardState = this.entryPoint

        for (layer in this.layers) {

            layer.forward(withinBatch, previousLayerState.numberOutputColumns, previousLayerState.forwardResult, isTraining)

            previousLayerState = layer

        }

        return previousLayerState.forwardResult

    }

}