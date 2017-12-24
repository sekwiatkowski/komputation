package com.komputation.cpu.layers.continuation.activation

import com.komputation.cpu.functions.activation.differentiateSigmoid
import com.komputation.cpu.functions.activation.sigmoid
import com.komputation.cpu.functions.hadamard
import com.komputation.cpu.layers.BaseCpuContinuationLayer
import com.komputation.cpu.layers.VariableLengthFloatArray

class CpuSigmoid internal constructor(
    name : String? = null,
    numberRows: Int,
    minimumColumns : Int,
    maximumColumns : Int) : BaseCpuContinuationLayer(name, numberRows, numberRows, minimumColumns, maximumColumns), CpuActivation {

    private val differentiationStore = VariableLengthFloatArray(numberRows, this.possibleOutputLengths)

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, forwardResult: FloatArray, isTraining: Boolean) {
        sigmoid(input, forwardResult, forwardResult.size)
    }

    /*
        input = pre-activation
        output = activation

        d activation / d pre-activation = activation * (1 - activation)
     */
    override fun computeBackwardResult(withinBatch: Int, numberInputColumns: Int, numberOutputColumns : Int, forwardResult : FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        val differentiation = this.differentiationStore.get(numberInputColumns)

        differentiateSigmoid(this.forwardResult, differentiation, differentiation.size)

        hadamard(chain, differentiation, backwardResult, backwardResult.size)
    }

}