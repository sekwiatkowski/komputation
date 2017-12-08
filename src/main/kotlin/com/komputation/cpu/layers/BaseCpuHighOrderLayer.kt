package com.komputation.cpu.layers

abstract class BaseCpuHigherOrderLayer(
    val name : String?,
    private val firstLayer : CpuForwardLayer,
    private val lastLayer : CpuForwardLayer) : CpuForwardLayer {

    override val numberOutputRows
        get() = this.lastLayer.numberOutputRows
    override val numberOutputColumns
        get() = this.lastLayer.numberOutputColumns
    override val forwardResult
        get() = this.lastLayer.forwardResult

    override val numberInputRows
        get() = this.firstLayer.numberInputRows
    override val numberInputColumns
        get() = this.firstLayer.numberInputColumns
    override val backwardResult
        get() = this.firstLayer.backwardResult

}