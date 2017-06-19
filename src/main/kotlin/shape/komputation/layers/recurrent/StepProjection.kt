package shape.komputation.layers.recurrent

import shape.komputation.functions.backwardProjectionWrtInput
import shape.komputation.functions.backwardProjectionWrtWeights
import shape.komputation.functions.project
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.*

class StepProjection(
    name : String? = null,
    private val weights : DoubleArray,
    private val numberWeightRows: Int,
    private val numberWeightColumns: Int,
    private val seriesAccumulator : DenseAccumulator? = null) : FeedForwardLayer(name) {

    private var input : DoubleMatrix? = null

    private val numberWeightEntries = numberWeightRows * numberWeightColumns

    override fun forward(input: DoubleMatrix) : DoubleMatrix {

        this.input = input

        val projected = project(input.entries, input.numberRows, input.numberColumns, weights, numberWeightRows, numberWeightColumns)

        return DoubleMatrix(numberWeightRows, input.numberColumns, projected)

    }

    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        val input = this.input!!
        val numberInputRows = input.numberRows
        val numberInputColumns = input.numberColumns

        val chainEntries = chain.entries
        val numberChainRows = chain.numberRows

        val gradient = backwardProjectionWrtInput(
            numberInputRows,
            numberInputColumns,
            numberInputRows * numberInputColumns,
            weights,
            numberWeightRows,
            chainEntries,
            numberChainRows)

        if (seriesAccumulator != null) {

            val stepDifferentiation = backwardProjectionWrtWeights(
                numberWeightEntries,
                numberWeightRows,
                numberWeightColumns,
                input.entries,
                numberInputRows,
                chainEntries,
                numberChainRows,
                chain.numberColumns)

            seriesAccumulator.accumulate(stepDifferentiation)

        }

        return DoubleMatrix(numberInputRows, numberInputColumns, gradient)

    }

}

fun createStepProjection(
    name : String?,
    weights: DoubleArray,
    numberWeightRows: Int,
    numberWeightColumns: Int,
    seriesAccumulator: DenseAccumulator? = null): StepProjection {

    return StepProjection(name, weights, numberWeightRows, numberWeightColumns, seriesAccumulator)

}
