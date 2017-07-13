package shape.komputation.cpu.layers.forward.projection

import shape.komputation.cpu.layers.BaseForwardLayer
import shape.komputation.cpu.optimization.DenseAccumulator
import shape.komputation.cpu.optimization.UpdateRule
import shape.komputation.cpu.optimization.updateDensely
import shape.komputation.functions.backwardProjectionWrtBias
import shape.komputation.functions.backwardProjectionWrtInput
import shape.komputation.functions.backwardProjectionWrtWeights
import shape.komputation.functions.project
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.Optimizable

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