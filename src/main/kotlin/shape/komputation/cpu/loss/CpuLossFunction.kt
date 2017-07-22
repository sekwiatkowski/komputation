package shape.komputation.cpu.loss

import shape.komputation.matrix.FloatMatrix

interface CpuLossFunction {

    fun forward(predictions: FloatMatrix, targets : FloatMatrix): Float

    fun backward(predictions: FloatMatrix, targets : FloatMatrix): FloatMatrix

}