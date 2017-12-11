package com.komputation.cpu.loss

class CpuSquaredLoss(
    numberInputRows: Int,
    minimumLength: Int,
    maximumLength : Int) : BaseCpuLossFunction(numberInputRows, minimumLength, maximumLength) {

    override fun computeLoss(targets: FloatArray, predictions: FloatArray): Float {
        var loss = 0f

        for (indexRow in 0 until targets.size) {
            val prediction = predictions[indexRow]
            val target = targets[indexRow]

            val difference = prediction - target

            loss += 0.5f * (difference * difference)
        }

        return loss
    }

    // loss = 0.5 (prediction - target)^2 = 0.5 prediction^2 - prediction * target + 0.5 target ^2
    // d loss / d prediction = prediction - target
    override fun computeDifferentation(targets: FloatArray, predictions: FloatArray, result : FloatArray) {
        for (indexEntry in 0 until targets.size) {
            result[indexEntry] = predictions[indexEntry] - targets[indexEntry]
        }
    }

}