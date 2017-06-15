package shape.komputation.layers.feedforward

import shape.komputation.functions.project
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeMatrix
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix
import shape.komputation.optimization.OptimizationStrategy
import shape.komputation.optimization.UpdateRule
import shape.komputation.optimization.updateDensely

fun createSharedProjectionLayer(
    name : String?,
    maximumSteps: Int,
    previousLayerRows: Int,
    nextLayerRows: Int,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null): SharedProjectionLayer {

    val weights = initializeMatrix(initializationStrategy, nextLayerRows, previousLayerRows)

    val weightUpdateRule : UpdateRule?

    if (optimizationStrategy != null) {

        weightUpdateRule = optimizationStrategy(weights.numberRows(), weights.numberColumns())

    }
    else {

        weightUpdateRule = null

    }

    return SharedProjectionLayer(name, maximumSteps, previousLayerRows, 1, weights, weightUpdateRule)

}


class SharedProjectionLayer(
    name : String? = null,
    maximumSteps : Int,
    private val numberInputRows : Int,
    private val numberInputColumns : Int,
    private val weights : RealMatrix,
    private val weightUpdateRule: UpdateRule? = null) : FeedForwardLayer(name), OptimizableLayer {

    private val optimize = weightUpdateRule != null

    private var inputs = arrayOfNulls<RealMatrix>(maximumSteps)
    private var step = -1

    private var seriesDifferentiation: DoubleArray? = null

    private val numberInputEntries = numberInputRows * numberInputColumns

    private val numberWeightRows = weights.numberRows()
    private val numberWeightColumns = weights.numberColumns()
    private val numberWeightEntries = numberWeightRows * numberWeightColumns

    fun resetForward() {

        step = -1

    }

    fun resetBackward() {

        if (optimize) {

            seriesDifferentiation = DoubleArray(numberWeightEntries)

        }

    }

    override fun forward(input: RealMatrix) : RealMatrix {

        step++

        inputs[step] = input

        return project(input, weights)

    }

    override fun backward(chain : RealMatrix) : RealMatrix {

        val input = inputs[step]!!

        val chainEntries = chain.getEntries()
        val numberChainRows = chain.numberRows()

        val gradient = differentiateProjectionWrtInput(numberInputRows, numberInputColumns, numberInputEntries, weights.getEntries(), numberWeightRows, chainEntries, numberChainRows)

        if (optimize) {

            val stepDifferentiation = differentiateProjectionWrtWeights(numberWeightRows, numberWeightColumns, numberWeightEntries, input.getEntries(), numberInputRows, chainEntries, numberChainRows, chain.numberColumns())

            val entries = stepDifferentiation.getEntries()
            val seriesDifferentiation = this.seriesDifferentiation!!

            var index = 0
            for (entry in entries) {

                seriesDifferentiation[index++] += entry

            }

        }

        step--

        return gradient

    }

    override fun optimize() {

        if (optimize) {

            updateDensely(this.weights.getEntries(), seriesDifferentiation!!, weightUpdateRule!!)

        }

    }

}