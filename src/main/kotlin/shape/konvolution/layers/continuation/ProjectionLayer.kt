package shape.konvolution.layers.continuation

import shape.konvolution.*
import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix
import shape.konvolution.matrix.project
import shape.konvolution.optimization.Optimizer

class ProjectionLayer(
    private val weights : RealMatrix,
    private val bias : RealMatrix? = null,
    private val weightOptimizer: Optimizer? = null,
    private val biasOptimizer: Optimizer? = null) : ContinuationLayer, OptimizableContinuationLayer {

    private val optimizeWeights = weightOptimizer != null
    private val optimizeBias = biasOptimizer != null

    override fun forward(input: RealMatrix) =

        arrayOf(project(input, weights, bias))

    override fun backward(inputs: Array<RealMatrix>, outputs : Array<RealMatrix>, chain : RealMatrix) : BackwardResult {

        val input = inputs.last()

        val inputGradient = differentiateProjectionWrtInput(input.numberRows(), input.numberColumns(), weights, chain)

        val parameterGradients = differentiateProjectionWrtParameters(input, weights, optimizeWeights, bias, optimizeBias, chain)

        return BackwardResult(inputGradient, parameterGradients)

    }

    override fun optimize(gradients: Array<RealMatrix?>) {

        if (optimizeWeights) {

            val weightGradient = gradients.first()

            this.weightOptimizer!!.optimize(this.weights, weightGradient!!)

        }

        if (optimizeBias) {

            val biasGradient = gradients.last()

            this.biasOptimizer!!.optimize(this.bias!!, biasGradient!!)

        }

    }

}

fun createProjectionLayer(
    previousLayerRows: Int,
    nextLayerRows: Int,
    initializationStrategy : () -> Double,
    weightOptimizer: Optimizer,
    biasOptimizer: Optimizer): ProjectionLayer {

    val weights = initializeMatrix(initializationStrategy, nextLayerRows, previousLayerRows)
    val bias = initializeRowVector(initializationStrategy, nextLayerRows)

    return ProjectionLayer(weights, bias, weightOptimizer, biasOptimizer)

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


fun differentiateProjectionWrtParameters(input: RealMatrix, weights: RealMatrix, optimizeWeights : Boolean, bias : RealMatrix?, optimizeBias: Boolean, chain: RealMatrix) : Array<RealMatrix?>? =

    if (optimizeWeights && optimizeBias) {

        val weightGradient = differentiateProjectionWrtWeights(weights.numberRows(), weights.numberColumns(), input, chain)
        val biasGradient = differentiateProjectionWrtBias(bias!!.numberRows(), chain)

        arrayOf<RealMatrix?>(weightGradient, biasGradient)

    }
    else if (optimizeWeights) {

        val weightGradient = differentiateProjectionWrtWeights(weights.numberRows(), weights.numberColumns(), input, chain)

        arrayOf<RealMatrix?>(weightGradient, null)

    }
    else if (optimizeBias) {

        val biasGradient = differentiateProjectionWrtBias(bias!!.numberRows(), chain)

        arrayOf<RealMatrix?>(null, biasGradient)

    }
    else {

        null

    }


fun differentiateProjectionWrtWeights(numberWeightRows : Int, numberWeightColumns: Int, input: RealMatrix, chain: RealMatrix) =

    createRealMatrix(numberWeightRows, numberWeightColumns).let { derivatives ->

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

        derivatives

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
