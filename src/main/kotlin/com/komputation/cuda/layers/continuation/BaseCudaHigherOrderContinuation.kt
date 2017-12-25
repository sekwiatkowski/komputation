package com.komputation.cuda.layers.continuation

import com.komputation.cuda.CudaBackwardState
import com.komputation.cuda.CudaForwardState
import com.komputation.cuda.layers.CudaContinuation
import jcuda.Pointer

abstract class BaseCudaHigherOrderContinuation(
    override val name : String?,
    private val firstLayer: CudaBackwardState,
    private val lastLayer: CudaForwardState) : CudaContinuation {

    override val numberOutputRows: Int
        get() = this.lastLayer.numberOutputRows
    override val maximumOutputColumns: Int
        get() = this.lastLayer.maximumOutputColumns
    override val deviceForwardResult: Pointer
        get() = this.lastLayer.deviceForwardResult
    override val deviceForwardLengths: Pointer
        get() = this.lastLayer.deviceForwardLengths
    override val batchMaximumOutputColumns: Int
        get() = this.lastLayer.batchMaximumOutputColumns

    override val numberInputRows: Int
        get() = this.firstLayer.numberInputRows
    override val maximumInputColumns: Int
        get() = this.firstLayer.maximumInputColumns
    override val deviceBackwardResult: Pointer
        get() = this.firstLayer.deviceBackwardResult
    override val batchMaximumInputColumns: Int
        get() = this.firstLayer.batchMaximumInputColumns

}