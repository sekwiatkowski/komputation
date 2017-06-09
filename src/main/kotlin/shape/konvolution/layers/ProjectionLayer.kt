package shape.konvolution.layers

import no.uib.cipr.matrix.Matrix
import shape.konvolution.*
import shape.konvolution.optimization.Optimizable
import shape.konvolution.optimization.Optimizer

class ProjectionLayer(
    private val weights : Matrix,
    private val bias : Matrix? = null,
    private val weightOptimizer: Optimizer? = null,
    private val biasOptimizer: Optimizer? = null) : Layer, Optimizable {

    private val optimizeWeights = weightOptimizer != null
    private val optimizeBias = biasOptimizer != null

    override fun forward(input: Matrix) =

        project(input, weights, bias)

    override fun backward(input: Matrix, output : Matrix, chain : Matrix) : BackwardResult {

        val inputGradient = differentiateProjectionWrtInput(input, weights, chain)

        val parameterGradients = differentiateProjectionWrtParameters(input, weights, optimizeWeights, bias, optimizeBias, chain)

        return BackwardResult(inputGradient, parameterGradients)

    }

    override fun optimize(gradients: Array<Matrix?>) {

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

fun differentiateProjectionWrtInput(input: Matrix, weights: Matrix, chain: Matrix) =

    createDenseMatrix(input.numRows(), input.numColumns()).let { derivatives ->

        for (indexInputRow in 0..input.numRows() - 1) {

            for (indexInputColumn in 0..input.numColumns() - 1) {

                var derivative = 0.0

                for (indexWeightRow in 0..weights.numRows() - 1) {

                    val chainEntry = chain.get(indexWeightRow, indexInputColumn)
                    val weightEntry = weights.get(indexWeightRow, indexInputRow)

                    derivative += chainEntry * weightEntry

                }

                derivatives.set(indexInputRow, indexInputColumn, derivative)

            }
        }

        derivatives

    }


fun differentiateProjectionWrtParameters(input: Matrix, weights: Matrix, optimizeWeights : Boolean, bias : Matrix?, optimizeBias: Boolean, chain: Matrix) : Array<Matrix?>? =

    if (optimizeWeights && optimizeBias) {

        val weightGradient = differentiateProjectionWrtWeights(input, weights, chain)
        val biasGradient = differentiateProjectionWrtBias(bias!!, chain)

        arrayOf<Matrix?>(weightGradient, biasGradient)

    }
    else if (optimizeWeights) {

        val weightGradient = differentiateProjectionWrtWeights(input, weights, chain)

        arrayOf<Matrix?>(weightGradient, null)

    }
    else if (optimizeBias) {

        val biasGradient = differentiateProjectionWrtBias(bias!!, chain)

        arrayOf<Matrix?>(null, biasGradient)

    }
    else {

        null

    }


fun differentiateProjectionWrtWeights(input: Matrix, weights : Matrix, chain: Matrix) =

    createDenseMatrix(weights.numRows(), weights.numColumns()).let { derivatives ->

        for (indexWeightRow in 0..weights.numRows() - 1) {

            for (indexWeightColumn in 0..weights.numColumns() - 1) {

                var derivative = 0.0

                for (indexChainColumn in 0..chain.numColumns() - 1) {

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

fun differentiateProjectionWrtBias(bias : Matrix, chain: Matrix) =

    createDenseMatrix(bias.numRows(), 1).let { derivatives ->

        for (indexRow in 0..bias.numRows() - 1) {

            var derivative = 0.0

            for (indexChainColumn in 0..chain.numColumns() - 1) {

                derivative += chain.get(indexRow, indexChainColumn)

            }

            derivatives.set(indexRow, 0, derivative)

        }

        derivatives

    }
