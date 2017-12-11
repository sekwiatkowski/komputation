package com.komputation.cpu.layers

abstract class BaseCpuHigherOrderLayer(
    val name : String?,
    private val firstLayer : CpuVariableLengthBackwardState,
    private val lastLayer : CpuVariableLengthForwardState) : CpuForwardLayer {

    override val numberOutputRows
        get() = this.lastLayer.numberOutputRows
    override val numberOutputColumns
        get() = this.lastLayer.numberOutputColumns
    override val possibleOutputLengths
        get() = this.lastLayer.possibleOutputLengths
    override val forwardResult
        get() = this.lastLayer.forwardResult

    override val numberInputRows
        get() = this.firstLayer.numberInputRows
    override val numberInputColumns
        get() = this.firstLayer.numberInputColumns
    override val possibleInputLengths
        get() = this.firstLayer.possibleInputLengths
    override val backwardResult
        get() = this.firstLayer.backwardResult

}