package shape.komputation.cpu.layers.forward.projection

import shape.komputation.cpu.functions.addBias
import shape.komputation.cpu.functions.backwardProjectionWrtBias
import shape.komputation.cpu.layers.BaseCpuVariableLengthForwardLayer
import shape.komputation.cpu.optimization.DenseAccumulator
import shape.komputation.cpu.optimization.UpdateRule
import shape.komputation.cpu.optimization.updateDensely
import shape.komputation.optimization.Optimizable

class CpuBiasLayer internal constructor(
    name : String? = null,
    numberRows : Int,
    minimumColumns: Int,
    maximumColumns: Int,
    private val bias : FloatArray,
    private val biasAccumulator: DenseAccumulator,
    private val biasUpdateRule: UpdateRule? = null) : BaseCpuVariableLengthForwardLayer(name, numberRows, numberRows, minimumColumns, maximumColumns), Optimizable {

    private val numberBiasEntries = numberRows

    override fun computeNumberOutputColumns(lengthIndex : Int, length: Int) = length

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean, result: FloatArray) {

        addBias(input, this.numberInputRows, numberInputColumns, this.bias, result)

    }

    override fun computeBackwardResult(withinBatch: Int, chain: FloatArray, result: FloatArray) {

        backwardProjectionWrtBias(this.numberBiasEntries, chain, this.numberOutputRows, this.numberOutputColumns, result)

        this.biasAccumulator.accumulate(result)

    }

    override fun optimize(batchSize : Int) {

        if (this.biasUpdateRule != null) {

            updateDensely(this.bias, this.biasAccumulator.getAccumulation(), this.numberBiasEntries, batchSize, this.biasUpdateRule)

            this.biasAccumulator.reset()

        }

    }

}