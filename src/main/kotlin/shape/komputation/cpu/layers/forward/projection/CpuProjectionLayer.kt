package shape.komputation.cpu.layers.forward.projection

import shape.komputation.cpu.functions.backwardProjectionWrtBias
import shape.komputation.cpu.functions.backwardProjectionWrtInput
import shape.komputation.cpu.functions.backwardProjectionWrtWeights
import shape.komputation.cpu.functions.project
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.optimization.DenseAccumulator
import shape.komputation.cpu.optimization.UpdateRule
import shape.komputation.cpu.optimization.updateDensely
import shape.komputation.matrix.FloatMatrix
import shape.komputation.optimization.Optimizable

class CpuProjectionLayer internal constructor(
    name : String? = null,

    private val weights : FloatArray,
    private val numberWeightRows: Int,
    private val numberWeightColumns: Int,
    private val weightAccumulator : DenseAccumulator,
    private val weightUpdateRule: UpdateRule? = null,

    private val bias : FloatArray? = null,
    private val biasAccumulator: DenseAccumulator? = null,
    private val biasUpdateRule: UpdateRule? = null) : BaseCpuForwardLayer(name), Optimizable {

    private var inputEntries = FloatArray(0)
    private var numberInputRows = -1
    private var numberInputColumns = -1

    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

    private val hasBias = bias != null
    private val numberBiasEntries = if(this.hasBias) this.numberWeightRows else 0
    private val backwardWrtBias = if(this.hasBias) FloatArray(bias!!.size) else null

    override fun forward(input: FloatMatrix, isTraining : Boolean) : FloatMatrix {

        this.inputEntries = input.entries.copyOf()
        this.numberInputRows = input.numberRows
        this.numberInputColumns = input.numberColumns

        val projection = project(this.inputEntries, this.numberInputRows, this.numberInputColumns, this.weights, this.numberWeightRows, this.numberWeightColumns, this.bias)

        return FloatMatrix(this.numberWeightRows, this.numberInputColumns, projection)

    }

    override fun backward(chain : FloatMatrix) : FloatMatrix {

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
            this.numberWeightEntries,
            this.numberWeightRows,
            this.numberWeightColumns,
            this.inputEntries,
            this.numberInputRows,
            chainEntries,
            numberChainRows,
            numberChainColumns)

        this.weightAccumulator.accumulate(backwardWrtWeights)

        if (this.biasAccumulator != null) {

            backwardProjectionWrtBias(this.bias!!.size, chainEntries, numberChainRows, numberChainColumns, this.backwardWrtBias!!)

            this.biasAccumulator.accumulate(this.backwardWrtBias)

        }

        return FloatMatrix(this.numberInputRows, this.numberInputColumns, backwardWrtInput)

    }

    override fun optimize(scalingFactor : Float) {

        if (this.weightUpdateRule != null) {

            val weightAccumulator = this.weightAccumulator

            updateDensely(this.weights, weightAccumulator.getAccumulation(), this.numberWeightEntries, scalingFactor, this.weightUpdateRule)

            weightAccumulator.reset()

        }

        if (hasBias) {

            val biasAccumulator = this.biasAccumulator!!

            updateDensely(this.bias!!, biasAccumulator.getAccumulation(), this.numberBiasEntries, scalingFactor, this.biasUpdateRule!!)

            biasAccumulator.reset()

        }

    }

}