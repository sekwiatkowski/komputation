package com.komputation.cpu.layers.forward.projection

import com.komputation.cpu.functions.addBias
import com.komputation.cpu.functions.backwardProjectionWrtBias
import com.komputation.cpu.layers.BaseCpuVariableLengthForwardLayer
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.cpu.optimization.UpdateRule
import com.komputation.cpu.optimization.updateDensely
import com.komputation.optimization.Optimizable

class CpuBiasLayer internal constructor(
    name : String? = null,
    numberRows : Int,
    minimumColumns: Int,
    maximumColumns: Int,
    private val bias : FloatArray,
    private val biasAccumulator: DenseAccumulator,
    private val biasUpdateRule: UpdateRule? = null) : BaseCpuVariableLengthForwardLayer(name, numberRows, numberRows, minimumColumns, maximumColumns), Optimizable {

    private val numberBiasEntries = numberRows

    fun getBias() = this.bias

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean, forwardResult: FloatArray) {
        addBias(input, this.numberInputRows, numberInputColumns, this.bias, forwardResult)
    }

    override fun computeBackwardResult(withinBatch: Int, numberInputColumns: Int, numberOutputColumns : Int, forwardResult : FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        backwardProjectionWrtBias(this.numberBiasEntries, chain, this.numberOutputRows, this.numberOutputColumns, backwardResult)

        this.biasAccumulator.accumulate(backwardResult)
    }

    override fun optimize(batchSize : Int) {
        if (this.biasUpdateRule != null) {
            updateDensely(this.bias, this.biasAccumulator.getAccumulation(), this.numberBiasEntries, batchSize, this.biasUpdateRule)

            this.biasAccumulator.reset()
        }
    }

}