package shape.konvolution.loss

import no.uib.cipr.matrix.Matrix

interface LossFunction {

    fun forward(predictions: Matrix, targets : Matrix): Double

    fun backward(predictions: Matrix, targets : Matrix): Matrix

}