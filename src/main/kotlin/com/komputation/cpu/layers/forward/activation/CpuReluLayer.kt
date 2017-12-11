package com.komputation.cpu.layers.forward.activation

import com.komputation.cpu.functions.activation.backwardRelu
import com.komputation.cpu.functions.activation.relu
import com.komputation.cpu.layers.BaseCpuForwardLayer

class CpuReluLayer internal constructor(name : String? = null, numberRows : Int, minimumColumns : Int, maximumColumns : Int) :
    BaseCpuForwardLayer(name, numberRows, numberRows, minimumColumns, maximumColumns), CpuActivationLayer {

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, forwardResult: FloatArray, isTraining: Boolean) {
        relu(input, forwardResult, forwardResult.size)
    }

    override fun computeBackwardResult(withinBatch: Int, numberInputColumns: Int, numberOutputColumns: Int, forwardResult : FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        backwardRelu(forwardResult, chain, backwardResult, backwardResult.size)
    }

}