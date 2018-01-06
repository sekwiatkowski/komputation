package com.komputation.demos.total

import com.komputation.matrix.FloatMatrix
import com.komputation.matrix.floatMatrix
import java.util.*

object TotalData {

    fun generateFixedLengthInput(random: Random, length: Int, from : Int, to : Int, numberExamples : Int): Array<FloatMatrix> {
        val inputRange = to - from

        return Array(numberExamples) {
            val numbers = FloatArray(length) { (random.nextInt(inputRange) + from).toFloat() }

            floatMatrix(1, length, *numbers)
        }
    }

    fun generateVariableLengthInput(random: Random, minimumLength: Int, maximumLength: Int, from : Int, to : Int, numberExamples : Int): Array<FloatMatrix> {
        val lengthRange = maximumLength - minimumLength
        val inputRange = to - from

        return Array(numberExamples) {
            val length = random.nextInt(lengthRange) + minimumLength

            val input = FloatArray(length) { (random.nextInt(inputRange) + from).toFloat() }

            floatMatrix(1, length, *input)
        }
    }

    fun generateTargets(input : Array<FloatMatrix>) =
        Array(input.size) { indexTarget ->
            floatArrayOf(input[indexTarget].entries.sum())
        }

}