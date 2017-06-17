package shape.komputation.loss

import shape.komputation.matrix.DoubleMatrix

interface LossFunction {

    fun forward(predictions: DoubleMatrix, targets : DoubleMatrix): Double

    fun backward(predictions: DoubleMatrix, targets : DoubleMatrix): DoubleMatrix

}