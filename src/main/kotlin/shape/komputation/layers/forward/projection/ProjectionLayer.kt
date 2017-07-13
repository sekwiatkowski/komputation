package shape.komputation.layers.forward.projection

import shape.komputation.functions.backwardProjectionWrtBias
import shape.komputation.functions.backwardProjectionWrtInput
import shape.komputation.functions.backwardProjectionWrtWeights
import shape.komputation.functions.project
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeColumnVector
import shape.komputation.initialization.initializeWeights
import shape.komputation.layers.BaseForwardLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.*

class CpuProjectionLayer internal constructor(
    name : String? = null,

    private val weights : DoubleArray,
    private val numberWeightRows: Int,
    private val numberWeightColumns: Int,
    private val weightAccumulator : DenseAccumulator,
    private val weightUpdateRule: UpdateRule? = null,

    private val bias : DoubleArray? = null,
    private val biasAccumulator: DenseAccumulator? = null,
    private val biasUpdateRule: UpdateRule? = null) : BaseForwardLayer(name), Optimizable {

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

        val backwardWrtInput = backwardProjectionWrtInput(
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

        return DoubleMatrix(this.numberInputRows, this.numberInputColumns, backwardWrtInput)

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

class ProjectionLayer(
    private val name : String?,
    private val inputDimension: Int,
    private val outputDimension: Int,
    private val weightInitializationStrategy: InitializationStrategy,
    private val biasInitializationStrategy: InitializationStrategy?,
    private val optimizationStrategy : OptimizationStrategy? = null) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuProjectionLayer {

        val numberWeightRows = this.outputDimension
        val numberWeightColumns = this.inputDimension

        val weights = initializeWeights(this.weightInitializationStrategy, numberWeightRows, numberWeightColumns, this.inputDimension)
        val weightUpdateRule = this.optimizationStrategy?.invoke(numberWeightRows, numberWeightColumns)

        val bias : DoubleArray?
        val biasUpdateRule: UpdateRule?
        val biasAccumulator: DenseAccumulator?

        if (this.biasInitializationStrategy != null) {

            bias = initializeColumnVector(this.biasInitializationStrategy, this.outputDimension)
            biasUpdateRule = this.optimizationStrategy?.invoke(bias.size, 1)
            biasAccumulator = DenseAccumulator(bias.size)

        }
        else {

            bias = null
            biasUpdateRule = null
            biasAccumulator = null

        }

        val weightAccumulator = DenseAccumulator(numberWeightRows * numberWeightColumns)

        return CpuProjectionLayer(this.name, weights, numberWeightRows, numberWeightColumns, weightAccumulator, weightUpdateRule, bias, biasAccumulator, biasUpdateRule)

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
    optimizationStrategy : OptimizationStrategy? = null) =

    ProjectionLayer(
        name,
        inputDimension,
        outputDimension,
        weightInitializationStrategy,
        biasInitializationStrategy,
        optimizationStrategy)