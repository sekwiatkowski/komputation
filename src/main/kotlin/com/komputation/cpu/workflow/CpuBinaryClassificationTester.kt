package com.komputation.cpu.workflow

class CpuBinaryTester(private val length : Int) : CpuClassificationTester {

    override fun test(predictions : FloatArray, targets : FloatArray) : Boolean {

        var isCorrect = true
        var step = 0

        while(step < this.length) {

            val prediction = predictions[step]
            val target = targets[step]

            isCorrect = (prediction < 0.5 && target == 0.0f) || (prediction >= 0.5 && target == 1.0f)

            if (isCorrect) {

                step++

            }
            else {

                break

            }

        }

        return isCorrect

    }

}

