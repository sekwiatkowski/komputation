package shape.komputation.layers.continuation

import shape.komputation.*
import shape.komputation.functions.project
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix
import shape.komputation.optimization.OptimizationStrategy
import shape.komputation.optimization.UpdateRule
import shape.komputation.optimization.updateDensely

class ProjectionLayer(
    name : String? = null,
    private val weights : RealMatrix,
    private val bias : RealMatrix? = null,
    private val weightUpdateRule: UpdateRule? = null,
    private val biasUpdateRule: UpdateRule? = null) : ContinuationLayer(name), OptimizableContinuationLayer {

    private val optimize = weightUpdateRule != null || biasUpdateRule != null

    private var forwardResult : RealMatrix? = null
    private var input : RealMatrix? = null

    private var backpropagationWrtWeights : RealMatrix? = null
    private var backpropagationWrtBias : RealMatrix? = null

    override fun forward(input: RealMatrix) : RealMatrix {

        this.input = input
        this.forwardResult = project(input, weights, bias)

        return this.forwardResult!!

    }

    override fun backward(chain : RealMatrix) : RealMatrix {

        val input = this.input!!

        val gradient = differentiateProjectionWrtInput(input.numberRows(), input.numberColumns(), weights, chain)

        if (optimize) {

            this.backpropagationWrtWeights = differentiateProjectionWrtWeights(weights.numberRows(), weights.numberColumns(), input, chain)

            if (bias != null) {

                this.backpropagationWrtBias = differentiateProjectionWrtBias(bias.numberRows(), chain)

            }

        }

        return gradient

    }

    override fun optimize() {

        if (optimize) {

            if (weightUpdateRule != null) {

                updateDensely(this.weights, this.backpropagationWrtWeights!!, weightUpdateRule)

            }

            if (bias != null && biasUpdateRule != null) {

                updateDensely(this.bias, this.backpropagationWrtBias!!, biasUpdateRule)

            }

        }

    }

}

fun createProjectionLayer(
    previousLayerRows: Int,
    nextLayerRows: Int,
    initializationStrategy : () -> Double,
    optimizationStrategy : OptimizationStrategy? = null) =

    createProjectionLayer(null, previousLayerRows, nextLayerRows, initializationStrategy, optimizationStrategy)

fun createProjectionLayer(
    name : String?,
    previousLayerRows: Int,
    nextLayerRows: Int,
    initializationStrategy : () -> Double,
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

    return ProjectionLayer(name, weights, bias, weightUpdateRule, biasUpdateRule)

}

fun differentiateProjectionWrtInput(numberInputRows: Int, numberInputColumns : Int, weights: RealMatrix, chain: RealMatrix) =

    createRealMatrix(numberInputRows, numberInputColumns).let { derivatives ->

        for (indexInputRow in 0..numberInputRows - 1) {

            for (indexInputColumn in 0..numberInputColumns - 1) {

                var derivative = 0.0

                for (indexWeightRow in 0..weights.numberRows() - 1) {

                    val chainEntry = chain.get(indexWeightRow, indexInputColumn)
                    val weightEntry = weights.get(indexWeightRow, indexInputRow)

                    derivative += chainEntry * weightEntry

                }

                derivatives.set(indexInputRow, indexInputColumn, derivative)

            }
        }

        derivatives

    }


fun differentiateProjectionWrtWeights(numberWeightRows : Int, numberWeightColumns: Int, input: RealMatrix, chain: RealMatrix): RealMatrix {

    val derivatives = createRealMatrix(numberWeightRows, numberWeightColumns)

    for (indexWeightRow in 0..numberWeightRows - 1) {

        for (indexWeightColumn in 0..numberWeightColumns - 1) {

            var derivative = 0.0

            for (indexChainColumn in 0..chain.numberColumns() - 1) {

                // d loss / d pre1, d loss / d pre2
                // All multiplications on other rows equal to zero
                val chainEntry = chain.get(indexWeightRow, indexChainColumn)

                // d pre ij / d wk
                val inputEntry = input.get(indexWeightColumn, indexChainColumn)

                derivative += chainEntry * inputEntry

            }

            derivatives.set(indexWeightRow, indexWeightColumn, derivative)

        }
    }

    return derivatives

}

fun differentiateProjectionWrtBias(numberBiasRows : Int, chain: RealMatrix) =

    createRealMatrix(numberBiasRows, 1).let { derivatives ->

        for (indexRow in 0..numberBiasRows - 1) {

            var derivative = 0.0

            for (indexChainColumn in 0..chain.numberColumns() - 1) {

                derivative += chain.get(indexRow, indexChainColumn)

            }

            derivatives.set(indexRow, 0, derivative)

        }

        derivatives

    }
