package com.komputation.cuda.layers

import com.komputation.cuda.CudaForwardState
import com.komputation.instructions.Resourceful
import jcuda.Pointer

abstract class BaseCudaEntryPoint(
    override val name: String?,
    final override val maximumOutputColumns : Int,
    final override val numberOutputRows: Int) : CudaEntryPoint, CudaForwardState, Resourceful {

    private val maximumBatchSizeArrray = intArrayOf(-1)
    protected var maximumBatchSize
        get() = this.maximumBatchSizeArrray[0]
        set(value) { this.maximumBatchSizeArrray[0] = value }
    protected val pointerToMaximumBatchSize = Pointer.to(this.maximumBatchSizeArrray)

    override val maximumOutputEntries = this.maximumOutputColumns * this.numberOutputRows
    protected var maximumBatchOutputEntries = -1

    override fun acquire(maximumBatchSize: Int) {
        this.maximumBatchSize = maximumBatchSize
        this.maximumBatchOutputEntries = maximumBatchSize * this.maximumOutputEntries
    }

    override fun release() {
        this.maximumBatchSize = -1
        this.maximumBatchOutputEntries = -1
    }

}