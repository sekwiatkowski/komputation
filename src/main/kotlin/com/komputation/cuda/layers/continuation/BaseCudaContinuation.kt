package com.komputation.cuda.layers.continuation

import com.komputation.cuda.layers.CudaContinuation
import com.komputation.instructions.Resourceful
import jcuda.Pointer

abstract class BaseCudaContinuation(
    final override val name: String?,
    final override val numberInputRows : Int,
    final override val numberOutputRows : Int,
    final override val maximumInputColumns : Int,
    final override val maximumOutputColumns : Int) : CudaContinuation, Resourceful {

    protected val pointerToNumberOutputRows = Pointer.to(intArrayOf(this.numberOutputRows))
    final override val maximumOutputEntries = this.numberOutputRows * this.maximumOutputColumns
    protected val pointerToMaximumOutputEntries = Pointer.to(intArrayOf(this.maximumOutputEntries))

    protected val pointerToNumberInputRows = Pointer.to(intArrayOf(this.numberInputRows))
    protected val pointerToMaximumInputColumns = Pointer.to(intArrayOf(this.maximumInputColumns))

    final override val maximumInputEntries = this.numberInputRows * this.maximumInputColumns
    protected val pointerToMaximumInputEntries = Pointer.to(intArrayOf(this.maximumInputEntries))

    protected val forwardResultSizeArray = intArrayOf(-1)
    protected var forwardResultSize
        get() = this.forwardResultSizeArray[0]
        set(value) { this.forwardResultSizeArray[0] = value }
    protected val pointerToForwardResultSize = Pointer.to(this.forwardResultSizeArray)

    protected val backwardResultSizeArray = intArrayOf(-1)
    protected var backwardResultSize
        get() = this.backwardResultSizeArray[0]
        set(value) { this.backwardResultSizeArray[0] = value }
    protected val pointerToBackwardResultSize = Pointer.to(this.backwardResultSizeArray)

    protected var maximumBatchSize = 0

    protected var maximumInputColumnsInCompleteBatch = -1
    protected var maximumOutputColumnsInCompleteBatch = -1

    override fun acquire(maximumBatchSize: Int) {
        this.maximumBatchSize = maximumBatchSize
        this.maximumInputColumnsInCompleteBatch = this.maximumBatchSize * this.maximumInputColumns
        this.maximumOutputColumnsInCompleteBatch = this.maximumBatchSize * this.maximumOutputColumns

        this.forwardResultSize = this.maximumBatchSize * this.maximumOutputEntries
        this.backwardResultSize = this.maximumBatchSize * this.maximumInputEntries
    }

    override fun release() {
        this.maximumBatchSize = -1
        this.maximumInputColumnsInCompleteBatch = -1
        this.maximumOutputColumnsInCompleteBatch = -1

        this.forwardResultSize = -1
        this.backwardResultSize = -1
    }

}