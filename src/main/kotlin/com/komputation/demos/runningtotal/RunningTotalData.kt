package com.komputation.demos.runningtotal

import com.komputation.matrix.FloatMatrix
import com.komputation.matrix.floatMatrix
import java.util.*

object RunningTotalData {

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
            val vector = input[indexTarget].entries

            var runningSum = 0f

            FloatArray(vector.size) { indexEntry ->
                runningSum += vector[indexEntry]
                runningSum
            }
        }

    fun generateReversedTargets(input : Array<FloatMatrix>) =
        Array(input.size) { indexTarget ->
            val inputEntries = input[indexTarget].entries
            val reversedInputEntries = inputEntries.reversedArray()

            var runningSum = 0f

            val target = FloatArray(reversedInputEntries.size) { indexEntry ->
                runningSum += reversedInputEntries[indexEntry]
                runningSum
            }

            val reversedTarget = target.reversedArray()

            reversedTarget
        }

}