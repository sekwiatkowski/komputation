package com.komputation.cuda.layers.continuation.maxpooling

import com.komputation.cuda.allocateDeviceIntMemory
import com.komputation.cuda.computeDeviceIntArraySize
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeRowwiseLaunchConfiguration
import com.komputation.cuda.layers.continuation.BaseCudaVariableNumberColumnsContinuation
import com.komputation.cuda.setIntArray
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class CudaMaxPooling internal constructor(
    name: String?,
    numberRows: Int,
    maximumInputColumns: Int,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val warpSize: Int,
    private val maximumNumberThreadsPerBlock: Int) : BaseCudaVariableNumberColumnsContinuation(name, numberRows, numberRows, maximumInputColumns, { 1 }) {

    private var forwardKernel: Kernel? = null
    private var backwardKernel: Kernel? = null

    private val deviceMaxIndices = Pointer()
    private val pointerToMaxIndices = Pointer.to(this.deviceMaxIndices)

    private var forwardConfiguration = computeRowwiseLaunchConfiguration(this.numberInputRows, this.maximumInputColumns, this.warpSize, this.maximumNumberThreadsPerBlock)
    private val numberWarps = (this.maximumInputColumns+this.warpSize-1)/this.warpSize
    private val forwardSharedMemoryBytes = computeDeviceIntArraySize(this.numberWarps).toInt()

    override val deviceForwardLengths = Pointer()

    override fun acquire(maximumBatchSize : Int) {
        super.acquire(maximumBatchSize)

        this.forwardKernel = this.createForwardKernel()
        this.backwardKernel = this.createBackwardKernel()

        allocateDeviceIntMemory(this.deviceMaxIndices, maximumBatchSize * this.numberInputRows)
        setIntArray(IntArray(this.maximumBatchSize) { 1 }, this.maximumBatchSize, this.deviceForwardLengths)
    }

    override fun release() {
        super.release()

        this.forwardKernel!!.destroy()
        this.backwardKernel!!.destroy()

        cudaFree(this.deviceMaxIndices)
        cudaFree(this.deviceForwardLengths)
    }

    private var pointerToInputLengths = Pointer()

    private var isTraining = false
    override fun computeForwardResult(batchSize: Int, deviceInput: Pointer, deviceInputLengths : Pointer, isTraining: Boolean) {
        this.isTraining = isTraining
        this.pointerToInputLengths = Pointer.to(deviceInputLengths)

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToInputLengths,
                this.pointerToMaximumInputEntries,
                this.pointerToMaxIndices,
                Pointer.to(deviceInput),
                this.pointerToForwardResult
            ),
            this.maximumBatchSize,
            this.forwardConfiguration.numberBlocks,
            this.forwardConfiguration.numberThreadsPerBlock,
            this.forwardSharedMemoryBytes
        )

    }

    override fun computeBackwardResult(batchSize: Int, chain: Pointer) {
        this.backwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToInputLengths,
                this.pointerToMaximumInputEntries,
                this.pointerToNumberInputRows,
                this.pointerToMaxIndices,
                Pointer.to(chain),
                this.pointerToBackwardResult
            ),
            this.maximumBatchSize,
            this.numberInputRows,
            this.maximumInputColumns,
            0
        )
    }

}