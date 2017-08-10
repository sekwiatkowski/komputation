package shape.komputation.cpu.loss

import shape.komputation.cpu.layers.BackwardLayerState

interface CpuLossFunction : BackwardLayerState {

    fun forward(predictions: FloatArray, targets : FloatArray): Float

    fun backward(predictions: FloatArray, targets : FloatArray) : FloatArray

}