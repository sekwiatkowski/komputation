package shape.komputation.loss

import shape.komputation.matrix.DoubleMatrix

class SquaredLoss : LossFunction {

    override fun forward(predictions: DoubleMatrix, targets : DoubleMatrix): Double {

        var loss = 0.0

        val predictionEntries = predictions.entries
        val targetEntries = targets.entries

        for (indexRow in 0..predictionEntries.size-1) {

            val prediction = predictionEntries[indexRow]
            val target = targetEntries[indexRow]

            loss += 0.5 * Math.pow(prediction - target, 2.0)

        }

        return loss

    }

    // loss = 0.5 (prediction - target)^2 = 0.5 prediction^2 - prediction * target + 0.5 target ^2
    // d loss / d prediction = prediction - target

    override fun backward(predictions: DoubleMatrix, targets : DoubleMatrix): DoubleMatrix {

        val predictionEntries = predictions.entries
        val targetEntries = targets.entries

        val backwardEntries = DoubleArray(predictionEntries.size) { index ->

            predictionEntries[index] - targetEntries[index]

        }

        return DoubleMatrix(predictions.numberRows, predictions.numberColumns, backwardEntries)

    }

}

fun squaredLoss() = SquaredLoss()