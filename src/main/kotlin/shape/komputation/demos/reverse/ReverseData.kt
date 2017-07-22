package shape.komputation.demos.reverse

import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.floatZeroMatrix
import shape.komputation.matrix.oneHotArray
import java.util.*

object ReverseData {

    fun generateInputs(random : Random, numberExamples : Int, seriesLength : Int, numberCategories  : Int) =

        Array<Matrix>(numberExamples) {

            val input = floatZeroMatrix(numberCategories, seriesLength)

            for (indexStep in 0..seriesLength - 1) {

                input.setColumn(indexStep, oneHotArray(numberCategories, random.nextInt(10), 1.0f))

            }

            input

        }

    fun generateTargets(inputs : Array<Matrix>, seriesLength: Int, numberCategories: Int) =

        Array(inputs.size) { index ->

            val matrix = inputs[index] as FloatMatrix

            val reversedSequenceMatrix = floatZeroMatrix(numberCategories, seriesLength)

            for (indexStep in 0..seriesLength - 1) {

                val reverseStep = seriesLength - indexStep - 1

                val originalStep = matrix.getColumn(reverseStep).entries

                reversedSequenceMatrix.setColumn(indexStep, originalStep)
            }

            reversedSequenceMatrix

        }

}