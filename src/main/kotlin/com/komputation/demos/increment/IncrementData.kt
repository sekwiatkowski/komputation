package com.komputation.demos.increment

import com.komputation.matrix.FloatMatrix
import com.komputation.matrix.floatMatrix
import java.util.*

object IncrementData {

    fun generateInput(random: Random, length: Int, from : Int, to : Int, numberExamples : Int): Array<FloatMatrix> {
        val inputRange = to - from

        return Array(numberExamples) {
            val numbers = FloatArray(length) { (random.nextInt(inputRange) + from).toFloat() }

            floatMatrix(1, length, *numbers)
        }
    }

    fun generateTargets(input : Array<FloatMatrix>, increment : Int = 1) =
        Array(input.size) { indexTarget ->
            val vector = input[indexTarget].entries

            FloatArray(vector.size) { indexEntry -> vector[indexEntry] + increment }
        }

}