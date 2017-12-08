package com.komputation.cpu.loss

import com.komputation.matrix.FloatMath

class CpuLogisticLoss(
    maximumLength: Int,
    hasFixedLength : Boolean) : BaseCpuLossFunction(2, maximumLength, hasFixedLength) {

    override fun computeLoss(targets: FloatArray, predictions: FloatArray): Float {
        var loss = 0.0f

        for (index in 0 until targets.size) {
            val target = targets[index]

            val prediction = predictions[index]

            if (target == 1.0f) {
                // -log(probability)
                loss += -FloatMath.log(prediction)
            }
            else {
                // -log(1 - probability)
                val counterProbability = 1.0f.minus(prediction)

                loss += -FloatMath.log(counterProbability)
            }
        }

        return loss
    }

    override fun computeDifferentation(targets: FloatArray, predictions: FloatArray, result: FloatArray) {
        for(indexEntry in 0 until targets.size) {

            val prediction = predictions[indexEntry]

            if (targets[indexEntry] == 1.0f) {

                result[indexEntry] = (-1.0f).div(prediction)

            }
            else {

                result[indexEntry] = 1.0f.div(1.0f - prediction)

            }
        }
    }

}