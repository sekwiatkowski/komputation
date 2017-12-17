package com.komputation.cpu.layers

abstract class BaseCpuHigherOrderLayer(
    val name : String?,
    override val numberInputRows : Int,
    override val numberOutputRows : Int,
    private val firstLayer : CpuVariableLengthBackwardState,
    private val lastLayer : CpuVariableLengthForwardState) : CpuForwardLayer {

    override val numberOutputColumns
        get() = this.lastLayer.numberOutputColumns
    override val possibleOutputLengths
        get() = this.lastLayer.possibleOutputLengths
    override val forwardResult
        get() = this.lastLayer.forwardResult

    override val numberInputColumns
        get() = this.firstLayer.numberInputColumns
    override val possibleInputLengths
        get() = this.firstLayer.possibleInputLengths
    override val backwardResult
        get() = this.firstLayer.backwardResult

}