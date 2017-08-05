package shape.komputation.demos.reverse

import shape.komputation.cpu.functions.getStep
import shape.komputation.cpu.functions.setStep
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.floatZeroMatrix
import shape.komputation.matrix.oneHotArray
import java.util.*

object ReverseData {

    fun generateInputs(random : Random, numberExamples : Int, seriesLength : Int, numberCategories  : Int) =

        Array<Matrix>(numberExamples) {

            val input = floatZeroMatrix(numberCategories, seriesLength)
            val inputEntries = input.entries

            for (indexStep in 0..seriesLength - 1) {

                setStep(oneHotArray(numberCategories, random.nextInt(10), 1.0f), indexStep, inputEntries, numberCategories)

            }

            input

        }

    fun generateTargets(inputs : Array<Matrix>, seriesLength: Int, numberCategories: Int) =

        Array(inputs.size) { index ->

            val matrix = inputs[index] as FloatMatrix

            val reversedSequenceMatrix = floatZeroMatrix(numberCategories, seriesLength)
            val reversedSequenceMatrixEntries = reversedSequenceMatrix.entries

            for (indexStep in 0..seriesLength - 1) {

                val reverseStep = seriesLength - indexStep - 1

                val originalStep = FloatArray(numberCategories)
                getStep(matrix.entries, reverseStep, originalStep, numberCategories)

                setStep(originalStep, indexStep, reversedSequenceMatrixEntries, numberCategories)
            }

            reversedSequenceMatrix

        }

}