package shape.komputation.demos.reverse

import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.doubleZeroMatrix
import shape.komputation.matrix.oneHotArray
import java.util.*

object ReverseData {

    fun generateInputs(random : Random, numberExamples : Int, seriesLength : Int, numberCategories  : Int) =

        Array<Matrix>(numberExamples) {

            val input = doubleZeroMatrix(numberCategories, seriesLength)

            for (indexStep in 0..seriesLength - 1) {

                input.setColumn(indexStep, oneHotArray(numberCategories, random.nextInt(10), 1.0))

            }

            input

        }

    fun generateTargets(inputs : Array<Matrix>, seriesLength: Int, numberCategories: Int) =

        Array<DoubleMatrix>(inputs.size) { index ->

            val matrix = inputs[index] as DoubleMatrix

            val reversedSequenceMatrix = doubleZeroMatrix(numberCategories, seriesLength)

            for (indexStep in 0..seriesLength - 1) {

                val reverseStep = seriesLength - indexStep - 1

                val originalStep = matrix.getColumn(reverseStep).entries

                reversedSequenceMatrix.setColumn(indexStep, originalStep)
            }

            reversedSequenceMatrix

        }

}