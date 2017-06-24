package shape.komputation.layers.feedforward.projection

import shape.komputation.functions.backwardProjectionWrtBias
import shape.komputation.functions.backwardProjectionWrtInput
import shape.komputation.functions.backwardProjectionWrtWeights
import shape.komputation.functions.project
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeMatrix
import shape.komputation.initialization.initializeRowVector
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.DenseAccumulator
import shape.komputation.optimization.OptimizationStrategy
import shape.komputation.optimization.UpdateRule
import shape.komputation.optimization.updateDensely

class ProjectionLayer(
    name : String? = null,
    private val weights : DoubleArray,
    private val numberWeightRows: Int,
    private val numberWeightColumns: Int,
    private val bias : DoubleArray? = null,
    private val weightUpdateRule: UpdateRule? = null,
    private val biasUpdateRule: UpdateRule? = null) : ContinuationLayer(name), OptimizableLayer {

    private var inputEntries = DoubleArray(0)
    private var numberInputRows = -1
    private var numberInputColumns = -1

    private val numberWeightEntries = numberWeightRows * numberWeightColumns
    private var weightAccumulator = DenseAccumulator(numberWeightRows * numberWeightColumns)

    private val numberBiasEntries = if(bias != null) bias.size else -1
    private val biasAccumulator = if(bias != null) DenseAccumulator(bias.size) else null

    override fun forward(input: DoubleMatrix) : DoubleMatrix {

        this.inputEntries = input.entries
        this.numberInputRows = input.numberRows
        this.numberInputColumns = input.numberColumns

        val projection = project(this.inputEntries, this.numberInputRows, this.numberInputColumns, this.weights, this.numberWeightRows, this.numberWeightColumns, this.bias)

        return DoubleMatrix(this.numberWeightRows, this.numberInputColumns, projection)

    }

    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        val chainEntries = chain.entries
        val numberChainRows = chain.numberRows
        val numberChainColumns = chain.numberColumns

        val numberInputEntries = this.numberInputRows * this.numberInputColumns

        val gradient = backwardProjectionWrtInput(
            this.numberInputRows,
            this.numberInputColumns,
            numberInputEntries,
            this.weights,
            this.numberWeightRows,
            chainEntries,
            numberChainRows)

        val backwardWrtWeights = backwardProjectionWrtWeights(
            numberWeightEntries,
            numberWeightRows,
            numberWeightColumns,
            this.inputEntries,
            this.numberInputRows,
            chainEntries,
            numberChainRows,
            numberChainColumns)

        this.weightAccumulator.accumulate(backwardWrtWeights)

        if (this.biasAccumulator != null) {

            val backwardWrtBias = backwardProjectionWrtBias(this.bias!!.size, chainEntries, numberChainRows, numberChainColumns)

            this.biasAccumulator.accumulate(backwardWrtBias)


        }

        return DoubleMatrix(this.numberInputRows, this.numberInputColumns, gradient)

    }

    override fun optimize() {

        if (this.weightUpdateRule != null) {

            val weightAccumulator = this.weightAccumulator

            updateDensely(this.weights, this.numberWeightEntries, weightAccumulator.getAccumulation(), weightAccumulator.getCount(), weightUpdateRule)

            weightAccumulator.reset()

        }

        if (this.bias != null && this.biasUpdateRule != null) {

            val biasAccumulator = this.biasAccumulator!!

            updateDensely(this.bias, this.numberBiasEntries, biasAccumulator.getAccumulation(), biasAccumulator.getCount(), biasUpdateRule)

            biasAccumulator.reset()

        }

    }

}

fun createProjectionLayer(
    numberInputRows: Int,
    numberResultRows: Int,
    withBias : Boolean,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null) =

    createProjectionLayer(null, numberInputRows, numberResultRows, withBias, initializationStrategy, optimizationStrategy)

fun createProjectionLayer(
    name : String?,
    inputDimension: Int,
    outputDimension: Int,
    withBias : Boolean,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null): ProjectionLayer {

    val numberWeightRows = outputDimension
    val numberWeightColumns = inputDimension

    val weights = initializeMatrix(initializationStrategy, numberWeightRows, numberWeightColumns)
    val weightUpdateRule = optimizationStrategy?.invoke(numberWeightRows, numberWeightColumns)

    val bias : DoubleArray?
    val biasUpdateRule: UpdateRule?

    if (withBias) {

        bias = initializeRowVector(initializationStrategy, numberWeightRows)
        biasUpdateRule = optimizationStrategy?.invoke(bias.size, 1)

    }
    else {

        bias = null
        biasUpdateRule = null

    }

    return ProjectionLayer(name, weights, numberWeightRows, numberWeightColumns, bias, weightUpdateRule, biasUpdateRule)

}