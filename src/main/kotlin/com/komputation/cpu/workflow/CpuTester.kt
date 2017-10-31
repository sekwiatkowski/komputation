package com.komputation.cpu.workflow

import com.komputation.cpu.network.CpuForwardPropagator
import com.komputation.matrix.Matrix

class CpuTester(
    private val forwardPropagator: CpuForwardPropagator,
    private val batches : Array<IntArray>,
    private val inputs : Array<Matrix>,
    private val targets: Array<FloatArray>,
    private val tester : CpuClassificationTester) {

    fun run(): Float {

        var numberPredictions = 0
        var numberCorrectPredictions = 0

        for(batch in this.batches) {

            for (index in batch) {

                val input = this.inputs[index]

                val predictions = this.forwardPropagator.forward(0, input, false)
                val targets = this.targets[index]

                val isCorrect = this.tester.test(predictions, targets)

                if(isCorrect) {

                    numberCorrectPredictions++

                }

                numberPredictions++

            }

        }

        return numberCorrectPredictions.toFloat().div(numberPredictions.toFloat())

    }

}