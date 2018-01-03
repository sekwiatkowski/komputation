package com.komputation.cuda.layers.continuation

import com.komputation.cuda.CudaBackwardState
import com.komputation.cuda.CudaForwardState
import jcuda.Pointer

abstract class BaseCudaHigherOrderContinuation(
    name : String?,
    private val firstLayer: CudaBackwardState,
    private val lastLayer: CudaForwardState) : BaseCudaContinuation(name, firstLayer.numberInputRows, lastLayer.numberOutputRows, firstLayer.maximumInputColumns, lastLayer.maximumOutputColumns) {

    override val deviceForwardResult: Pointer
        get() = this.lastLayer.deviceForwardResult
    override val deviceForwardLengths: Pointer
        get() = this.lastLayer.deviceForwardLengths
    override val largestNumberOutputColumnsInCurrentBatch: Int
        get() = this.lastLayer.largestNumberOutputColumnsInCurrentBatch

    override val deviceBackwardResult: Pointer
        get() = this.firstLayer.deviceBackwardResult
    override val largestNumberInputColumnsInCurrentBatch: Int
        get() = this.firstLayer.largestNumberInputColumnsInCurrentBatch

}