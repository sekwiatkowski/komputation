package shape.komputation.layers.feedforward.projection

import shape.komputation.functions.backwardProjectionWrtInput
import shape.komputation.functions.backwardProjectionWrtWeights
import shape.komputation.functions.project
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.*

class SharedProjectionLayer(
    name : String? = null,
    private val numberInputRows: Int,
    numberOutputRows: Int,
    private val weights : DoubleArray,
    private val seriesAccumulator : DenseAccumulator? = null) : FeedForwardLayer(name) {

    private var input : DoubleMatrix? = null

    private val numberInputColumns = 1
    private val numberInputEntries = numberInputRows * numberInputColumns

    private val numberWeightRows = numberOutputRows
    private val numberWeightColumns = numberInputRows
    private val numberWeightEntries = numberWeightRows * numberWeightColumns

    override fun forward(input: DoubleMatrix) : DoubleMatrix {

        this.input = input

        val projected = project(input.entries, numberInputRows, numberInputColumns, weights, numberWeightRows, numberWeightColumns)

        return DoubleMatrix(numberWeightRows, numberInputColumns, projected)

    }

    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        val input = this.input!!

        val chainEntries = chain.entries
        val numberChainRows = chain.numberRows

        val gradient = backwardProjectionWrtInput(
            numberInputRows,
            numberInputColumns,
            numberInputEntries,
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

fun createSharedProjectionLayer(
    numberInputRows: Int,
    numberOutputRows: Int,
    weights: DoubleArray,
    seriesAccumulator: DenseAccumulator? = null) =

    createSharedProjectionLayer(null, numberInputRows, numberOutputRows, weights, seriesAccumulator)

fun createSharedProjectionLayer(
    name : String?,
    numberInputRows: Int,
    numberOutputRows: Int,
    weights: DoubleArray,
    seriesAccumulator: DenseAccumulator? = null): SharedProjectionLayer {

    return SharedProjectionLayer(name, numberInputRows, numberOutputRows, weights, seriesAccumulator)

}
