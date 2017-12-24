package com.komputation.cuda.layers

import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.instructions.Resourceful
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

abstract class BaseCudaContinuation(
    val name: String?,
    final override val numberInputRows : Int,
    final override val numberOutputRows : Int,
    final override val maximumInputColumns : Int,
    final override val maximumOutputColumns : Int) : CudaContinuation, Resourceful {

    protected val pointerToNumberInputRows = Pointer.to(intArrayOf(this.numberInputRows))
    protected val pointerToNumberOutputRows = Pointer.to(intArrayOf(this.numberOutputRows))

    protected val maximumInputEntries
        get() = this.numberInputRows * this.maximumInputColumns
    protected val pointerToMaximumInputEntries = Pointer.to(intArrayOf(this.maximumInputEntries))

    protected val maximumOutputEntries
        get() = this.numberOutputRows * this.maximumOutputColumns
    protected val pointerToMaximumOutputEntries = Pointer.to(intArrayOf(this.maximumOutputEntries))

    protected var maximumBatchSize = 0

    protected var maximumBatchInputColumns = -1
    protected var maximumBatchOutputColumns = -1

    override var batchMaximumInputColumns = -1
    override var batchMaximumOutputColumns = -1

    final override val deviceForwardResult = Pointer()
    protected val pointerToForwardResult = Pointer.to(this.deviceForwardResult)

    protected var forwardResultSize = intArrayOf(-1)
    protected val pointerToForwardResultSize = Pointer.to(this.forwardResultSize)

    final override val deviceBackwardResult = Pointer()
    protected val pointerToBackwardResult = Pointer.to(this.deviceBackwardResult)

    protected var backwardResultSize = intArrayOf(-1)
    protected val pointerToBackwardResultSize = Pointer.to(this.backwardResultSize)

    protected val batchSize = intArrayOf(-1)
    protected val pointerToBatchSize = Pointer.to(this.batchSize)

    override fun acquire(maximumBatchSize: Int) {
        this.maximumBatchSize = maximumBatchSize
        this.maximumBatchInputColumns = this.maximumBatchSize * this.maximumInputColumns
        this.maximumBatchOutputColumns = this.maximumBatchSize * this.maximumOutputColumns

        this.forwardResultSize[0] = this.maximumBatchSize * this.maximumOutputEntries
        this.backwardResultSize[0] = this.maximumBatchSize * this.maximumInputEntries

        allocateDeviceFloatMemory(this.deviceForwardResult, this.forwardResultSize[0])
        allocateDeviceFloatMemory(this.deviceBackwardResult, this.backwardResultSize[0])
    }

    override fun release() {
        this.maximumBatchSize = -1
        this.maximumBatchInputColumns = -1
        this.maximumBatchOutputColumns = -1

        this.forwardResultSize[0] = -1
        this.backwardResultSize[0] = -1

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceBackwardResult)
    }

    override fun forward(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, batchMaximumInputLength: Int, isTraining: Boolean): Pointer {
        this.batchSize[0] = batchSize

        this.batchMaximumInputColumns = batchMaximumInputLength
        computeOutputLengths(deviceInputLengths, batchMaximumInputLength)
        computeForwardResult(batchSize, deviceInput, deviceInputLengths, batchMaximumInputLength, isTraining)

        return this.deviceForwardResult
    }

    abstract fun computeOutputLengths(deviceInputLengths: Pointer, batchMaximumInputLength: Int)

    protected abstract fun computeForwardResult(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, batchMaximumInputLength: Int, isTraining: Boolean)

    override fun backward(batchSize: Int, chain: Pointer) : Pointer {
        computeBackwardResult(batchSize, chain)

        return this.deviceBackwardResult
    }

    protected abstract fun computeBackwardResult(batchSize: Int, chain: Pointer)

}