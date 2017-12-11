package com.komputation.cpu.layers.forward.activation

import com.komputation.cpu.layers.BaseCpuForwardLayer

class CpuIdentityLayer internal constructor(
    name : String? = null,
    numberRows: Int,
    minimumColumns : Int,
    maximumColumns : Int) : BaseCpuForwardLayer(name, numberRows, numberRows, minimumColumns, maximumColumns, { inputLength -> inputLength }), CpuActivationLayer {

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, forwardResult: FloatArray, isTraining: Boolean) {
        System.arraycopy(input, 0, forwardResult, 0, input.size)
    }

    override fun computeBackwardResult(withinBatch: Int, numberInputColumns: Int, numberOutputColumns: Int, forwardResult: FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        System.arraycopy(chain, 0, backwardResult, 0, chain.size)
    }

}