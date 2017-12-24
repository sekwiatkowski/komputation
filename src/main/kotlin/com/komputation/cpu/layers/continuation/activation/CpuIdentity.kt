package com.komputation.cpu.layers.continuation.activation

import com.komputation.cpu.layers.BaseCpuContinuationLayer

class CpuIdentity internal constructor(
    name : String? = null,
    numberRows: Int,
    minimumColumns : Int,
    maximumColumns : Int) : BaseCpuContinuationLayer(name, numberRows, numberRows, minimumColumns, maximumColumns, { inputLength -> inputLength }), CpuActivation {

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, forwardResult: FloatArray, isTraining: Boolean) {
        System.arraycopy(input, 0, forwardResult, 0, input.size)
    }

    override fun computeBackwardResult(withinBatch: Int, numberInputColumns: Int, numberOutputColumns: Int, forwardResult: FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        System.arraycopy(chain, 0, backwardResult, 0, chain.size)
    }

}