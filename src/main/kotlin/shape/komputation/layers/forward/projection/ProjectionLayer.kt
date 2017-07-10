package shape.komputation.layers.forward.projection

import shape.komputation.functions.backwardProjectionWrtBias
import shape.komputation.functions.backwardProjectionWrtInput
import shape.komputation.functions.backwardProjectionWrtWeights
import shape.komputation.functions.project
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeColumnVector
import shape.komputation.initialization.initializeWeights
import shape.komputation.layers.ForwardLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.*

class ProjectionLayer internal constructor(
    name : String? = null,
    private val weights : DoubleArray,
    private val numberWeightRows: Int,
    private val numberWeightColumns: Int,
    private val weightAccumulator : DenseAccumulator,
    private val weightUpdateRule: UpdateRule? = null,

    private val bias : DoubleArray? = null,
    private val biasUpdateRule: UpdateRule? = null,
    private val biasAccumulator: DenseAccumulator? = null) : ForwardLayer(name), Optimizable {

    private var inputEntries = DoubleArray(0)
    private var numberInputRows = -1
    private var numberInputColumns = -1

    private val numberWeightEntries = numberWeightRows * numberWeightColumns

    override fun forward(input: DoubleMatrix, isTraining : Boolean) : DoubleMatrix {

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

    override fun optimize(scalingFactor : Double) {

        if (this.weightUpdateRule != null) {

            val weightAccumulator = this.weightAccumulator

            updateDensely(this.weights, weightAccumulator.getAccumulation(), scalingFactor, this.weightUpdateRule)

            weightAccumulator.reset()

        }

        if (this.bias != null && this.biasUpdateRule != null) {

            val biasAccumulator = this.biasAccumulator!!

            updateDensely(this.bias, biasAccumulator.getAccumulation(), scalingFactor, this.biasUpdateRule)

            biasAccumulator.reset()

        }

    }

}

fun projectionLayer(
    inputDimension: Int,
    outputDimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : OptimizationStrategy? = null) =

    projectionLayer(null, inputDimension, outputDimension, weightInitializationStrategy, biasInitializationStrategy, optimizationStrategy)

fun projectionLayer(
    name : String?,
    inputDimension: Int,
    outputDimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : OptimizationStrategy? = null): ProjectionLayer {

    val numberWeightRows = outputDimension
    val numberWeightColumns = inputDimension

    val weights = initializeWeights(weightInitializationStrategy, numberWeightRows, numberWeightColumns, inputDimension)
    val weightUpdateRule = optimizationStrategy?.invoke(numberWeightRows, numberWeightColumns)

    val bias : DoubleArray?
    val biasUpdateRule: UpdateRule?
    val biasAccumulator: DenseAccumulator?

    if (biasInitializationStrategy != null) {

        bias = initializeColumnVector(biasInitializationStrategy, outputDimension)
        biasUpdateRule = optimizationStrategy?.invoke(bias.size, 1)
        biasAccumulator = DenseAccumulator(bias.size)

    }
    else {

        bias = null
        biasUpdateRule = null
        biasAccumulator = null

    }

    val weightAccumulator = DenseAccumulator(numberWeightRows * numberWeightColumns)

    return ProjectionLayer(name, weights, numberWeightRows, numberWeightColumns, weightAccumulator, weightUpdateRule, bias, biasUpdateRule, biasAccumulator)

}