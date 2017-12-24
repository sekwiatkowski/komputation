package com.komputation.cpu.layers.continuation.activation

import com.komputation.cpu.layers.BaseCpuHigherOrderContinuation
import com.komputation.cpu.layers.continuation.normalization.CpuNormalization

class CpuSoftmax internal constructor(
    name : String? = null,
    private val exponentiation: CpuExponentiation,
    private val normalization: CpuNormalization) : BaseCpuHigherOrderContinuation(name, exponentiation, normalization), CpuActivation {

    override fun forward(withinBatch : Int, numberInputColumns : Int, input : FloatArray, isTraining : Boolean): FloatArray {
        this.exponentiation.forward(withinBatch, numberInputColumns, input, isTraining)

        val normalized = this.normalization.forward(withinBatch, this.exponentiation.numberOutputColumns, this.exponentiation.forwardResult, isTraining)

        return normalized
    }

    override fun backward(withinBatch : Int, chain : FloatArray): FloatArray {
        val backwardNormalization = this.normalization.backward(withinBatch, chain)

        val backwardExponentiation = this.exponentiation.backward(withinBatch, backwardNormalization)

        return backwardExponentiation
    }

}