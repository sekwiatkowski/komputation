package shape.komputation.cpu.loss

import shape.komputation.matrix.DoubleMatrix

interface CpuLossFunction {

    fun forward(predictions: DoubleMatrix, targets : DoubleMatrix): Double

    fun backward(predictions: DoubleMatrix, targets : DoubleMatrix): DoubleMatrix

}