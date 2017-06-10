package shape.konvolution.layers.continuation

import shape.konvolution.*
import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix
import shape.konvolution.matrix.project
import shape.konvolution.optimization.UpdateRule
import shape.konvolution.optimization.updateDensely

class ProjectionLayer(
    private val weights : RealMatrix,
    private val bias : RealMatrix? = null,
    private val weightUpdateRule: UpdateRule? = null,
    private val biasUpdateRule: UpdateRule? = null) : ContinuationLayer, OptimizableContinuationLayer {

    val optimize = weightUpdateRule != null || biasUpdateRule != null

    override fun forward(input: RealMatrix): RealMatrix =

        project(input, weights, bias)

    override fun backward(input: RealMatrix, output : RealMatrix, chain : RealMatrix) : BackwardResult {

        val inputGradient = differentiateProjectionWrtInput(input.numberRows(), input.numberColumns(), weights, chain)

        val parameterGradients = if(optimize) differentiateProjectionWrtParameters(input, weights, bias, chain) else null

        return BackwardResult(inputGradient, parameterGradients)

    }

    override fun optimize(gradients: Array<RealMatrix?>) {

        if (optimize) {

            if (weightUpdateRule != null) {

                val weightGradient = gradients.first()

                updateDensely(this.weights, weightGradient!!, weightUpdateRule)

            }

            if (bias != null && biasUpdateRule != null) {

                val biasGradient = gradients.last()

                updateDensely(this.bias, biasGradient!!, biasUpdateRule)

            }


        }

    }

}

fun createProjectionLayer(
    previousLayerRows: Int,
    nextLayerRows: Int,
    initializationStrategy : () -> Double,
    optimizationStrategy : ((numberRows : Int, numberColumns : Int) -> UpdateRule)? = null): ProjectionLayer {

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

    return ProjectionLayer(weights, bias, weightUpdateRule, biasUpdateRule)

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


fun differentiateProjectionWrtParameters(input: RealMatrix, weights: RealMatrix, bias : RealMatrix?, chain: RealMatrix) : Array<RealMatrix?>? {

    val weightGradient = differentiateProjectionWrtWeights(weights.numberRows(), weights.numberColumns(), input, chain)

    if (bias != null) {

        val biasGradient = differentiateProjectionWrtBias(bias.numberRows(), chain)

        return arrayOf<RealMatrix?>(weightGradient, biasGradient)

    }
    else {

        return arrayOf<RealMatrix?>(weightGradient, null)

    }

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
