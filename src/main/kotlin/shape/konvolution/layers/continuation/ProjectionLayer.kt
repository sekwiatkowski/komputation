package shape.konvolution.layers.continuation

import shape.konvolution.*
import shape.konvolution.optimization.Optimizable
import shape.konvolution.optimization.Optimizer

class ProjectionLayer(
    private val weights : RealMatrix,
    private val bias : RealMatrix? = null,
    private val weightOptimizer: Optimizer? = null,
    private val biasOptimizer: Optimizer? = null) : ContinuationLayer, Optimizable {

    private val optimizeWeights = weightOptimizer != null
    private val optimizeBias = biasOptimizer != null

    override fun forward(input: RealMatrix) =

        project(input, weights, bias)

    override fun backward(input: RealMatrix, output : RealMatrix, chain : RealMatrix) : BackwardResult {

        val inputGradient = differentiateProjectionWrtInput(input, weights, chain)

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
    val bias = initializeBias(initializationStrategy, nextLayerRows)

    return ProjectionLayer(weights, bias, weightOptimizer, biasOptimizer)

}

fun differentiateProjectionWrtInput(input: RealMatrix, weights: RealMatrix, chain: RealMatrix) =

    createRealMatrix(input.numberRows(), input.numberColumns()).let { derivatives ->

        for (indexInputRow in 0..input.numberRows() - 1) {

            for (indexInputColumn in 0..input.numberColumns() - 1) {

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

        val weightGradient = differentiateProjectionWrtWeights(input, weights, chain)
        val biasGradient = differentiateProjectionWrtBias(bias!!, chain)

        arrayOf<RealMatrix?>(weightGradient, biasGradient)

    }
    else if (optimizeWeights) {

        val weightGradient = differentiateProjectionWrtWeights(input, weights, chain)

        arrayOf<RealMatrix?>(weightGradient, null)

    }
    else if (optimizeBias) {

        val biasGradient = differentiateProjectionWrtBias(bias!!, chain)

        arrayOf<RealMatrix?>(null, biasGradient)

    }
    else {

        null

    }


fun differentiateProjectionWrtWeights(input: RealMatrix, weights : RealMatrix, chain: RealMatrix) =

    createRealMatrix(weights.numberRows(), weights.numberColumns()).let { derivatives ->

        for (indexWeightRow in 0..weights.numberRows() - 1) {

            for (indexWeightColumn in 0..weights.numberColumns() - 1) {

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

fun differentiateProjectionWrtBias(bias : RealMatrix, chain: RealMatrix) =

    createRealMatrix(bias.numberRows(), 1).let { derivatives ->

        for (indexRow in 0..bias.numberRows() - 1) {

            var derivative = 0.0

            for (indexChainColumn in 0..chain.numberColumns() - 1) {

                derivative += chain.get(indexRow, indexChainColumn)

            }

            derivatives.set(indexRow, 0, derivative)

        }

        derivatives

    }
