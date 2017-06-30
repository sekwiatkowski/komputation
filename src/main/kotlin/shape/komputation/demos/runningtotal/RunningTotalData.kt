package shape.komputation.demos.runningtotal

import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix
import java.util.*

object RunningTotalData {

    fun generateInputs(random : Random, numberExamples : Int, numberSteps : Int, exclusiveUpperLimit : Int) = Array<Matrix>(numberExamples) {

        DoubleMatrix(1, numberSteps, DoubleArray(numberSteps) { random.nextInt(exclusiveUpperLimit).toDouble() })

    }

    fun generateTargets(inputs : Array<Matrix>, numberSteps : Int) = Array<DoubleMatrix>(inputs.size) { indexExample ->

        val input = inputs[indexExample] as DoubleMatrix

        val targetEntries = input
            .entries
            .foldIndexed(arrayListOf<Double>()) { index, list, current ->

                list.add(list.getOrElse(index-1, { 0.0 }) + current)

                list

            }
            .toDoubleArray()

        DoubleMatrix(1, numberSteps, targetEntries)

    }


}