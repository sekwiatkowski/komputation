package com.komputation.cpu.loss

import com.komputation.matrix.FloatMath

class CpuCrossEntropyLoss(
    numberInputRows: Int,
    maximumLength: Int,
    hasFixedLength : Boolean) : BaseCpuLossFunction(numberInputRows, maximumLength, hasFixedLength) {

    override fun computeLoss(targets: FloatArray, predictions: FloatArray): Float {
        var loss = 0.0f

        for (index in 0 until targets.size) {
            val target = targets[index]

            if (target == 1.0f) {
                loss += -FloatMath.log(predictions[index])
            }
        }

        return loss
    }

    override fun computeDifferentation(targets: FloatArray, predictions: FloatArray, result: FloatArray) {
        for(indexEntry in 0 until targets.size) {
            if (targets[indexEntry] == 1.0f) {
                result[indexEntry] = -1.0f.div(predictions[indexEntry])
            }
            else {
                result[indexEntry] = 0f
            }
        }
    }

}