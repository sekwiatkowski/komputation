package com.komputation.cpu.layers.forward.activation

import com.komputation.cpu.functions.activation.backwardExponentiation
import com.komputation.cpu.functions.activation.exponentiate
import com.komputation.cpu.layers.BaseCpuVariableLengthForwardLayer

class CpuExponentiationLayer internal constructor(
    name : String? = null,
    numberRows : Int,
    minimumColumns : Int,
    maximumColumns : Int) : BaseCpuVariableLengthForwardLayer(name, numberRows, numberRows, minimumColumns, maximumColumns), CpuActivationLayer {

    override fun computeNumberOutputColumns(inputLength: Int) = inputLength

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean, forwardResult: FloatArray) {
        exponentiate(input, forwardResult, forwardResult.size)
    }

    override fun computeBackwardResult(withinBatch: Int, numberInputColumns: Int, numberOutputColumns : Int, forwardResult : FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        backwardExponentiation(forwardResult, chain, backwardResult, backwardResult.size)
    }

}