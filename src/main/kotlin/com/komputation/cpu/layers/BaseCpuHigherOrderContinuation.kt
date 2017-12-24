package com.komputation.cpu.layers

abstract class BaseCpuHigherOrderContinuation(
    val name : String?,
    private val firstLayer: CpuBackwardState,
    private val lastLayer: CpuForwardState) : CpuContinuation {

    override val numberInputRows: Int
        get() = this.firstLayer.numberInputRows
    override val possibleInputLengths
        get() = this.firstLayer.possibleInputLengths
    override val numberInputColumns
        get() = this.firstLayer.numberInputColumns
    override val backwardResult
        get() = this.firstLayer.backwardResult

    override val numberOutputRows: Int
        get() = this.lastLayer.numberOutputRows
    override val possibleOutputLengths
        get() = this.lastLayer.possibleOutputLengths
    override val numberOutputColumns
        get() = this.lastLayer.numberOutputColumns
    override val forwardResult
        get() = this.lastLayer.forwardResult

}