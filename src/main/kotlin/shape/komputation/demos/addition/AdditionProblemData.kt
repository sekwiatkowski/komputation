package shape.komputation.demos.addition

import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.doubleScalar
import shape.komputation.matrix.sequence
import java.util.*

object AdditionProblemData {

    fun generateInputs(size: Int, random: Random, length: Int) =

        Array<Matrix>(size) {

            val input = sequence(length) { doubleArrayOf(random.nextDouble(), 0.0) }

            val firstStep = random.nextInt(length)
            val secondStep = random.nextInt(length).let { candidate ->

                if (candidate == firstStep) {

                    if (firstStep == length - 1) {
                        firstStep - 1
                    } else {
                        firstStep + 1
                    }

                } else {

                    candidate
                }

            }

            input.setEntry(1, firstStep, 1.0)
            input.setEntry(1, secondStep, 1.0)

            input
        }

    fun generateTarget(inputs: Array<Matrix>) =

        Array(inputs.size) { indexInput ->

            val input = inputs[indexInput] as DoubleMatrix

            var solution = 0.0
            for (indexStep in 0..input.numberColumns - 1) {

                val step = input.getColumn(indexStep)

                solution += step.entries[0] * step.entries[1]

            }

            doubleScalar(solution)

        }
}