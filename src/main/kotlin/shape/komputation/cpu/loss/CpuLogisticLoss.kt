package shape.komputation.cpu.loss

import shape.komputation.matrix.DoubleMatrix

class CpuLogisticLoss : CpuLossFunction {

    // -log(probability of the correct target)
    override fun forward(predictions: DoubleMatrix, targets : DoubleMatrix): Double {

        val predictionEntries = predictions.entries
        val targetEntries = targets.entries

        var loss = 0.0

        for (index in 0..predictionEntries.size-1) {

            val target = targetEntries[index]

            if (target == 1.0) {

                loss -= Math.log(predictionEntries[index])

            }

        }

        return loss

    }

    // -1/target probability if target = 1.0, 0.0 otherwise
    override fun backward(predictions: DoubleMatrix, targets : DoubleMatrix) : DoubleMatrix {

        val predictionEntries = predictions.entries
        val targetEntries = targets.entries

        val derivatives = DoubleArray(predictionEntries.size) { index ->

            val target = targetEntries[index]

            if (target == 1.0) {

                -1.0.div(predictionEntries[index])

            }
            else {

                0.0

            }

        }

        return DoubleMatrix(predictions.numberRows, predictions.numberColumns, derivatives)

    }

}