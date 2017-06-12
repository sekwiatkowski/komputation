package shape.komputation.loss

import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix

class SquaredLoss : LossFunction {

    override fun forward(predictions: RealMatrix, targets : RealMatrix): Double {

        var loss = 0.0

        for (indexRow in 0..predictions.numberRows()-1) {

            for (indexColumn in 0..predictions.numberColumns()-1) {

                loss += 0.5 * Math.pow(predictions.get(indexRow, indexColumn) - targets.get(indexRow, indexColumn), 2.0)

            }

        }

        return loss

    }

    // loss = 0.5 (prediction - target)^2 = 0.5 prediction^2 - prediction * target + 0.5 target ^2
    // d loss / d prediction = prediction - target

    override fun backward(predictions: RealMatrix, targets : RealMatrix) : RealMatrix {

        val derivatives = createRealMatrix(predictions.numberRows(), predictions.numberColumns())

        for (indexRow in 0..predictions.numberRows()-1) {

            for (indexColumn in 0..predictions.numberColumns() - 1) {

                val derivative = predictions.get(indexRow, indexColumn) - targets.get(indexRow, indexColumn)

                derivatives.set(indexRow, indexColumn, derivative)

            }

        }

        return derivatives

    }

}