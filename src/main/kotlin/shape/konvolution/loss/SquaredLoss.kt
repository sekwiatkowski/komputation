package shape.konvolution.loss

import no.uib.cipr.matrix.Matrix
import shape.konvolution.createDenseMatrix

class SquaredLoss : LossFunction {

    override fun forward(predictions: Matrix, targets : Matrix): Double {

        var loss = 0.0

        for (indexRow in 0..predictions.numRows()-1) {

            for (indexColumn in 0..predictions.numColumns()-1) {

                loss += 0.5 * Math.pow(predictions.get(indexRow, indexColumn) - targets.get(indexRow, indexColumn), 2.0)

            }

        }

        return loss

    }

    // loss = 0.5 (prediction - target)^2 = 0.5 prediction^2 - prediction * target + 0.5 target ^2
    // d loss / d prediction = prediction - target

    override fun backward(predictions: Matrix, targets : Matrix) : Matrix {

        val derivatives = createDenseMatrix(predictions.numRows(), predictions.numColumns())

        for (indexRow in 0..predictions.numRows()-1) {

            for (indexColumn in 0..predictions.numColumns() - 1) {

                val derivative = predictions.get(indexRow, indexColumn) - targets.get(indexRow, indexColumn)

                derivatives.set(indexRow, indexColumn, derivative)

            }

        }

        return derivatives

    }

}