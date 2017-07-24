package shape.komputation.demos.addition

import shape.komputation.cpu.functions.getStep
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.floatScalar
import java.util.*

object AdditionProblemData {

    fun generateInputs(size: Int, random: Random, length: Int) =

        Array<Matrix>(size) {

            val input = FloatArray(length * 2)
            for (index in 0..length - 1) {

                input[index * 2] = random.nextFloat()

            }

            val firstStep = random.nextInt(length)
            val secondStep = random.nextInt(length).let { candidate ->

                if (candidate == firstStep) {

                    if (firstStep == length - 1) {

                        firstStep - 1

                    }
                    else {

                        firstStep + 1

                    }

                }
                else {

                    candidate

                }

            }

            input[firstStep * 2 + 1] = 1.0f
            input[secondStep * 2 + 1] = 1.0f

            FloatMatrix(2, length, input)

        }

    fun generateTarget(inputs: Array<Matrix>) =

        Array(inputs.size) { indexInput ->

            val input = inputs[indexInput] as FloatMatrix

            var solution = 0.0f

            val inputEntries = input.entries

            for (indexStep in 0..input.numberColumns - 1) {

                val step = FloatArray(2)
                getStep(inputEntries, indexStep, step, 2)

                solution += step[0] * step[1]

            }

            floatScalar(solution)

        }
}