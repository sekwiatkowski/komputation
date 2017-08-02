package shape.komputation.cpu.loss

import shape.komputation.matrix.FloatMath
import shape.komputation.matrix.FloatMatrix

class CpuLogisticLoss : CpuLossFunction {

    // -log(probability of the correct target)
    override fun forward(predictions: FloatMatrix, targets : FloatMatrix): Float {

        val predictionEntries = predictions.entries
        val targetEntries = targets.entries

        var loss = 0.0f

        for (index in 0..predictionEntries.size-1) {

            val target = targetEntries[index]

            if (target == 1.0f) {

                loss += -FloatMath.log(predictionEntries[index])

            }

        }

        return loss

    }

    // -1/target probability if target = 1.0, 0.0 otherwise
    override fun backward(predictions: FloatMatrix, targets : FloatMatrix) : FloatMatrix {

        val predictionEntries = predictions.entries
        val targetEntries = targets.entries

        val derivatives = FloatArray(predictionEntries.size) { index ->

            val target = targetEntries[index]

            if (target == 1.0f) {

                -1.0f.div(predictionEntries[index])

            }
            else {

                0.0f

            }

        }

        return FloatMatrix(predictions.numberRows, predictions.numberColumns, derivatives)

    }

}