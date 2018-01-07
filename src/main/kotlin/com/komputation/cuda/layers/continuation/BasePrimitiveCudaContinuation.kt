package com.komputation.cuda.layers.continuation

import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.layers.CudaContinuation
import com.komputation.instructions.Resourceful
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

abstract class BasePrimitiveCudaContinuation(
    name: String?,
    numberInputRows : Int,
    numberOutputRows : Int,
    maximumInputColumns : Int,
    maximumOutputColumns : Int) : BaseCudaContinuation(name, numberInputRows, numberOutputRows, maximumInputColumns, maximumOutputColumns), CudaContinuation, Resourceful {

    final override val deviceForwardResult = Pointer()
    protected val pointerToForwardResult = Pointer.to(this.deviceForwardResult)

    final override val deviceBackwardResult = Pointer()
    protected val pointerToBackwardResult = Pointer.to(this.deviceBackwardResult)

    private val batchSizeArray = intArrayOf(-1)
    protected var batchSize
        get() = this.batchSizeArray[0]
        set(value) { this.batchSizeArray[0] = value }
    protected val pointerToBatchSize = Pointer.to(this.batchSizeArray)

    override fun acquire(maximumBatchSize: Int) {
        super.acquire(maximumBatchSize)

        allocateDeviceFloatMemory(this.deviceForwardResult, this.forwardResultSize)
        allocateDeviceFloatMemory(this.deviceBackwardResult, this.backwardResultSize)
    }

    override fun release() {
        super.release()

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceBackwardResult)
    }

    override fun forward(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, isTraining: Boolean): Pointer {
        this.batchSize = batchSize

        computeOutputLengths(deviceInputLengths)
        computeForwardResult(batchSize, deviceInput, deviceInputLengths, isTraining)

        return this.deviceForwardResult
    }

    abstract fun computeOutputLengths(deviceInputLengths: Pointer)

    protected abstract fun computeForwardResult(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, isTraining: Boolean)

    override fun backward(batchSize: Int, chain: Pointer) : Pointer {
        computeBackwardResult(batchSize, chain)

        return this.deviceBackwardResult
    }

    protected abstract fun computeBackwardResult(batchSize: Int, chain: Pointer)

}