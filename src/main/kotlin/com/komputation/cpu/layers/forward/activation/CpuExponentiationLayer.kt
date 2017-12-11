package com.komputation.cpu.layers.forward.activation

import com.komputation.cpu.functions.activation.backwardExponentiation
import com.komputation.cpu.functions.activation.exponentiate
import com.komputation.cpu.layers.BaseCpuForwardLayer

class CpuExponentiationLayer internal constructor(
    name : String? = null,
    numberRows : Int,
    minimumColumns : Int,
    maximumColumns : Int) : BaseCpuForwardLayer(name, numberRows, numberRows, minimumColumns, maximumColumns), CpuActivationLayer {

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, forwardResult: FloatArray, isTraining: Boolean) {
        exponentiate(input, forwardResult, forwardResult.size)
    }

    override fun computeBackwardResult(withinBatch: Int, numberInputColumns: Int, numberOutputColumns : Int, forwardResult : FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        backwardExponentiation(forwardResult, chain, backwardResult, backwardResult.size)
    }

}