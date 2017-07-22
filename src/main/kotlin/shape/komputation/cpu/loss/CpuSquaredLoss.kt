package shape.komputation.cpu.loss

import shape.komputation.matrix.FloatMatrix

class CpuSquaredLoss : CpuLossFunction {

    override fun forward(predictions: FloatMatrix, targets : FloatMatrix): Float {

        var loss = 0.0f

        val predictionEntries = predictions.entries
        val targetEntries = targets.entries

        for (indexRow in 0..predictionEntries.size-1) {

            val prediction = predictionEntries[indexRow]
            val target = targetEntries[indexRow]

            val difference = prediction - target

            loss += 0.5f * (difference * difference)


        }

        return loss

    }

    // loss = 0.5 (prediction - target)^2 = 0.5 prediction^2 - prediction * target + 0.5 target ^2
    // d loss / d prediction = prediction - target

    override fun backward(predictions: FloatMatrix, targets : FloatMatrix): FloatMatrix {

        val predictionEntries = predictions.entries
        val targetEntries = targets.entries

        val backwardEntries = FloatArray(predictionEntries.size) { index ->

            predictionEntries[index] - targetEntries[index]

        }

        return FloatMatrix(predictions.numberRows, predictions.numberColumns, backwardEntries)

    }

}