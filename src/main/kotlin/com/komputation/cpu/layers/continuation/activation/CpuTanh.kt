package com.komputation.cpu.layers.continuation.activation

import com.komputation.cpu.functions.activation.differentiateTanh
import com.komputation.cpu.functions.activation.tanh
import com.komputation.cpu.functions.hadamard
import com.komputation.cpu.layers.BaseCpuContinuationLayer
import com.komputation.cpu.layers.VariableLengthFloatArray

class CpuTanh internal constructor(
    name: String? = null,
    numberRows : Int,
    minimumColumns : Int,
    maximumColumns : Int) : BaseCpuContinuationLayer(name, numberRows, numberRows, minimumColumns, maximumColumns), CpuActivation {

    private val differentiationStore = VariableLengthFloatArray(numberRows, this.possibleOutputLengths)

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, forwardResult: FloatArray, isTraining: Boolean) {
        tanh(input, forwardResult, forwardResult.size)
    }

    override fun computeBackwardResult(withinBatch: Int, numberInputColumns: Int, numberOutputColumns : Int, forwardResult : FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        val differentiation = this.differentiationStore.get(numberInputColumns)

        differentiateTanh(forwardResult, differentiation, differentiation.size)

        hadamard(chain, differentiation, backwardResult, backwardResult.size)
    }

}