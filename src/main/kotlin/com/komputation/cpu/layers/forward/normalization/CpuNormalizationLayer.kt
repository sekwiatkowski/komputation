package com.komputation.cpu.layers.forward.normalization

import com.komputation.cpu.functions.activation.backwardNormalization
import com.komputation.cpu.functions.activation.normalize
import com.komputation.cpu.layers.BaseCpuVariableLengthForwardLayer
import com.komputation.cpu.layers.VariableLengthFloatArray
import com.komputation.cpu.layers.forward.activation.CpuActivationLayer

/*
    a/(a+b+c)

    input entry = a
    forward entry = a/(a+b+c)
 */
class CpuNormalizationLayer internal constructor(
    name : String? = null,
    numberRows : Int,
    minimumColumns : Int,
    maximumColumns : Int) : BaseCpuVariableLengthForwardLayer(name, numberRows, numberRows, minimumColumns, maximumColumns), CpuActivationLayer {

    private var sumStore = VariableLengthFloatArray(numberRows, minimumColumns, this.possibleLengths, { inputLength -> computeNumberOutputColumns(inputLength) })
    private var sum = FloatArray(0)

    override fun computeNumberOutputColumns(inputLength: Int) = inputLength

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean, forwardResult: FloatArray) {
        this.sum = this.sumStore.get(numberInputColumns)

        normalize(this.numberInputRows, numberInputColumns, input, this.sum, forwardResult)
    }

    override fun computeBackwardResult(withinBatch: Int, numberInputColumns: Int, numberOutputColumns : Int, forwardResult : FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        backwardNormalization(this.numberInputRows, numberInputColumns, chain, forwardResult, this.sum, backwardResult)
    }

}