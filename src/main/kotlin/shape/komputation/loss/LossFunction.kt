package shape.komputation.loss

import shape.komputation.matrix.RealMatrix

interface LossFunction {

    fun forward(predictions: RealMatrix, targets : RealMatrix): Double

    fun backward(predictions: RealMatrix, targets : RealMatrix): RealMatrix

}