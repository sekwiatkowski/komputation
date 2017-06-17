package shape.komputation.layers.feedforward.projection

import shape.komputation.functions.backwardProjectionWrtInput
import shape.komputation.functions.backwardProjectionWrtWeights
import shape.komputation.functions.project
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeMatrix
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.StatefulLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.*

class SharedProjectionLayer(
    name : String? = null,
    private val numberInputRows: Int,
    numberOutputRows: Int,
    private val weights : DoubleArray,
    private val weightUpdateRule: UpdateRule? = null) : FeedForwardLayer(name), StatefulLayer, OptimizableLayer {

    private val optimize = weightUpdateRule != null

    private var input : DoubleMatrix? = null

    private val numberInputColumns = 1
    private val numberInputEntries = numberInputRows * numberInputColumns

    private val numberWeightRows = numberOutputRows
    private val numberWeightColumns = numberInputRows
    private val numberWeightEntries = numberWeightRows * numberWeightColumns

    private var seriesAccumulator = if(optimize) DenseAccumulator(numberWeightEntries) else null
    private var batchAccumulator = if(optimize) DenseAccumulator(numberWeightEntries) else null

    override fun startForward() {

    }

    override fun forward(input: DoubleMatrix) : DoubleMatrix {

        this.input = input

        val projected = project(input.entries, numberInputRows, numberInputColumns, weights, numberWeightRows, numberWeightColumns)

        return DoubleMatrix(numberWeightRows, numberInputColumns, projected)

    }

    override fun finishBackward() {

        if (optimize) {

            val seriesAccumulation = this.seriesAccumulator!!

            batchAccumulator!!.accumulate(seriesAccumulation.getAccumulation())

            seriesAccumulation.reset()

        }

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

        if (optimize) {

            val stepDifferentiation = backwardProjectionWrtWeights(
                numberWeightEntries,
                numberWeightRows,
                numberWeightColumns,
                input.entries,
                numberInputRows,
                chainEntries,
                numberChainRows,
                chain.numberColumns)

            seriesAccumulator!!.accumulate(stepDifferentiation)

        }

        return DoubleMatrix(numberInputRows, numberInputColumns, gradient)

    }

    override fun optimize() {

        if (optimize) {

            val batchAccumulator = this.batchAccumulator!!

            updateDensely(this.weights, batchAccumulator.getAccumulation(), batchAccumulator.getCount(), weightUpdateRule!!)

            this.batchAccumulator!!.reset()

        }

    }

}

fun createSharedProjectionLayer(
    numberInputRows: Int,
    numberOutputRows: Int,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null) =

    createSharedProjectionLayer(null, numberInputRows, numberOutputRows, initializationStrategy, optimizationStrategy)

fun createSharedProjectionLayer(
    name : String?,
    numberInputRows: Int,
    numberOutputRows: Int,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null): SharedProjectionLayer {

    val weights = initializeMatrix(initializationStrategy, numberOutputRows, numberInputRows)

    val weightUpdateRule : UpdateRule?

    if (optimizationStrategy != null) {

        weightUpdateRule = optimizationStrategy(numberOutputRows, numberInputRows)

    }
    else {

        weightUpdateRule = null

    }

    return SharedProjectionLayer(name, numberInputRows, numberOutputRows, weights, weightUpdateRule)

}
