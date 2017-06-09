package shape.konvolution.loss

import no.uib.cipr.matrix.Matrix
import shape.konvolution.createDenseMatrix

class LogisticLoss : LossFunction {

    // -log(probability of the correct target)
    override fun forward(predictions: Matrix, targets : Matrix): Double {

        var loss = 0.0

        for (indexColumn in 0..targets.numColumns() -1) {

            for (indexRow in 0..targets.numRows() - 1) {

                val target = targets.get(indexRow, indexColumn)

                if (target == 1.0) {

                    loss -= Math.log(predictions.get(indexRow, indexColumn))
                }

            }

        }

        return loss

    }

    // -1/probability of the correct target summed over each column
    override fun backward(predictions: Matrix, targets : Matrix) : Matrix {

        val derivatives = createDenseMatrix(predictions.numRows(), predictions.numColumns())

        for (indexColumn in 0..predictions.numColumns() -1) {

            for (indexRow in 0..targets.numRows() - 1) {

                val target = targets.get(indexRow, indexColumn)

                if (target == 1.0) {

                    val prediction = predictions.get(indexRow, indexColumn)

                    derivatives.set(indexRow, indexColumn, -1.0.div(prediction))

                }


            }

        }

        return derivatives

    }

}