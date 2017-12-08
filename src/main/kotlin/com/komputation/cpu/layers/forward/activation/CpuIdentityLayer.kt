package com.komputation.cpu.layers.forward.activation

import com.komputation.cpu.layers.CpuForwardLayer

class CpuIdentityLayer internal constructor(
    private val name : String? = null,
    override val numberOutputRows: Int) : CpuForwardLayer, CpuActivationLayer {

    override var forwardResult = FloatArray(0)
    override val numberOutputColumns
        get() = this.numberInputColumns
    override var backwardResult = FloatArray(0)
    override val numberInputRows = this.numberOutputRows
    override var numberInputColumns = -1

    override fun forward(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean): FloatArray {
        this.forwardResult = input
        this.numberInputColumns = numberInputColumns
        return this.forwardResult
    }

    override fun backward(withinBatch: Int, chain: FloatArray): FloatArray {
        this.backwardResult = chain
        return this.backwardResult
    }

}