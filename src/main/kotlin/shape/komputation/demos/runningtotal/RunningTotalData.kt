package shape.komputation.demos.runningtotal

import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.Matrix
import java.util.*

object RunningTotalData {

    fun generateInputs(random : Random, numberExamples : Int, numberSteps : Int, exclusiveUpperLimit : Int) = Array<Matrix>(numberExamples) {

        FloatMatrix(1, numberSteps, FloatArray(numberSteps) { random.nextInt(exclusiveUpperLimit).toFloat() })

    }

    fun generateTargets(inputs : Array<Matrix>, numberSteps : Int) = Array<FloatMatrix>(inputs.size) { indexExample ->

        val input = inputs[indexExample] as FloatMatrix

        val targetEntries = input
            .entries
            .foldIndexed(arrayListOf<Float>()) { index, list, current ->

                list.add(list.getOrElse(index-1, { 0.0f }) + current)

                list

            }
            .toFloatArray()

        FloatMatrix(1, numberSteps, targetEntries)

    }


}