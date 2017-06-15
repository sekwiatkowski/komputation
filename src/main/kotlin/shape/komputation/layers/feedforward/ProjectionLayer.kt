package shape.komputation.layers.feedforward

import shape.komputation.functions.project
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeMatrix
import shape.komputation.initialization.initializeRowVector
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix
import shape.komputation.optimization.OptimizationStrategy
import shape.komputation.optimization.UpdateRule
import shape.komputation.optimization.updateDensely

class ProjectionLayer(
    name : String? = null,
    private val numberInputRows: Int,
    private val numberInputColumns: Int,
    private val weights : RealMatrix,
    private val bias : RealMatrix? = null,
    private val weightUpdateRule: UpdateRule? = null,
    private val biasUpdateRule: UpdateRule? = null) : FeedForwardLayer(name), OptimizableLayer {

    private val optimize = weightUpdateRule != null || biasUpdateRule != null

    private var forwardResult : RealMatrix? = null
    private var input : RealMatrix? = null

    private val numberWeightRows = weights.numberRows()
    private val numberWeightColumns = weights.numberColumns()
    private val numberWeightEntries = numberWeightRows * numberWeightColumns

    private val numberInputEntries = numberInputRows * numberInputColumns

    private var backpropagationWrtWeights : RealMatrix? = null
    private var backpropagationWrtBias : RealMatrix? = null

    override fun forward(input: RealMatrix) : RealMatrix {

        this.input = input
        this.forwardResult = project(input, weights, bias)

        return this.forwardResult!!

    }

    override fun backward(chain : RealMatrix) : RealMatrix {

        val input = this.input!!

        val chainEntries = chain.getEntries()
        val numberChainRows = chain.numberRows()

        val gradient = differentiateProjectionWrtInput(numberInputRows, numberInputColumns, numberInputEntries, weights.getEntries(), numberWeightRows, chainEntries, numberChainRows)

        if (optimize) {

            this.backpropagationWrtWeights = differentiateProjectionWrtWeights(
                numberWeightRows,
                numberWeightColumns,
                numberWeightEntries,
                input.getEntries(),
                numberInputRows,
                chainEntries,
                numberChainRows,
                chain.numberColumns())

            if (bias != null) {

                this.backpropagationWrtBias = differentiateProjectionWrtBias(bias.numberRows(), chain)

            }

        }

        return gradient

    }

    override fun optimize() {

        if (optimize) {

            if (this.weightUpdateRule != null) {

                updateDensely(this.weights.getEntries(), this.backpropagationWrtWeights!!.getEntries(), weightUpdateRule)

            }

            if (this.bias != null && this.biasUpdateRule != null) {

                updateDensely(this.bias.getEntries(), this.backpropagationWrtBias!!.getEntries(), biasUpdateRule)

            }

        }

    }

}

fun createProjectionLayer(
    previousLayerRows: Int,
    nextLayerRows: Int,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null) =

    createProjectionLayer(null, previousLayerRows, nextLayerRows, initializationStrategy, optimizationStrategy)

fun createProjectionLayer(
    name : String?,
    previousLayerRows: Int,
    nextLayerRows: Int,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null): ProjectionLayer {

    val weights = initializeMatrix(initializationStrategy, nextLayerRows, previousLayerRows)
    val bias = initializeRowVector(initializationStrategy, nextLayerRows)

    val weightUpdateRule : UpdateRule?
    val biasUpdateRule : UpdateRule?

    if (optimizationStrategy != null) {

        weightUpdateRule = optimizationStrategy(weights.numberRows(), weights.numberColumns())
        biasUpdateRule = optimizationStrategy(bias.numberRows(), bias.numberColumns())

    }
    else {

        weightUpdateRule = null
        biasUpdateRule = null

    }

    return ProjectionLayer(name, previousLayerRows, 1, weights, bias, weightUpdateRule, biasUpdateRule)

}

fun differentiateProjectionWrtInput(
    numberInputRows: Int,
    numberInputColumns : Int,
    numberInputEntries : Int,
    weightEntries : DoubleArray,
    numberWeightRows : Int,
    chainEntries: DoubleArray,
    numberChainRows : Int): RealMatrix {

    val derivatives = DoubleArray(numberInputEntries)

    var index = 0

    for (indexInputColumn in 0..numberInputColumns - 1) {

        for (indexInputRow in 0..numberInputRows - 1) {

            var derivative = 0.0

            for (indexWeightRow in 0..numberWeightRows - 1) {

                val chainEntry = chainEntries[indexWeightRow + indexInputColumn * numberChainRows]
                val weightEntry = weightEntries[indexWeightRow + indexInputRow * numberWeightRows]

                derivative += chainEntry * weightEntry

            }

            derivatives[index++] = derivative

        }
    }

    return createRealMatrix(numberInputRows, numberInputColumns, derivatives)

}

fun differentiateProjectionWrtWeights(
    numberWeightRows : Int,
    numberWeightColumns: Int,
    numberWeightEntries : Int,
    inputEntries: DoubleArray,
    numberInputRows : Int,
    chainEntries: DoubleArray,
    numberChainRows: Int,
    numberChainColumns : Int): RealMatrix {

    val derivatives = DoubleArray(numberWeightEntries)

    var index = 0

    for (indexWeightColumn in 0..numberWeightColumns - 1) {

        for (indexWeightRow in 0..numberWeightRows - 1) {

            var derivative = 0.0

            for (indexChainColumn in 0..numberChainColumns - 1) {

                // d loss / d pre1, d loss / d pre2
                // All multiplications on other rows equal to zero
                val chainEntry = chainEntries[indexWeightRow + indexChainColumn * numberChainRows]

                // d pre ij / d wk
                val inputEntry = inputEntries[indexWeightColumn + indexChainColumn * numberInputRows]

                derivative += chainEntry * inputEntry

            }

            derivatives[index++] = derivative

        }
    }

    return createRealMatrix(numberWeightRows, numberWeightColumns, derivatives)

}

fun differentiateProjectionWrtBias(numberBiasRows : Int, chain: RealMatrix) =

    createRealMatrix(numberBiasRows, 1).let { derivatives ->

        val numberChainColumns = chain.numberColumns()

        for (indexRow in 0..numberBiasRows - 1) {

            var derivative = 0.0

            for (indexChainColumn in 0..numberChainColumns - 1) {

                derivative += chain.get(indexRow, indexChainColumn)

            }

            derivatives.set(indexRow, 0, derivative)

        }

        derivatives

    }
