package shape.konvolution.loss

import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix

class LogisticLoss : LossFunction {

    // -log(probability of the correct target)
    override fun forward(predictions: RealMatrix, targets : RealMatrix): Double {

        var loss = 0.0

        for (indexColumn in 0..targets.numberColumns() -1) {

            for (indexRow in 0..targets.numberRows() - 1) {

                val target = targets.get(indexRow, indexColumn)

                if (target == 1.0) {

                    loss -= Math.log(predictions.get(indexRow, indexColumn))
                }

            }

        }

        return loss

    }

    // -1/probability of the correct target summed over each column
    override fun backward(predictions: RealMatrix, targets : RealMatrix) : RealMatrix {

        val derivatives = createRealMatrix(predictions.numberRows(), predictions.numberColumns())

        for (indexColumn in 0..predictions.numberColumns() -1) {

            for (indexRow in 0..targets.numberRows() - 1) {

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