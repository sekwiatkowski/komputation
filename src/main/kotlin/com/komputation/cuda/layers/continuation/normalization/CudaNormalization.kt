package com.komputation.cuda.layers.continuation.normalization

import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.computeDeviceFloatArraySize
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import com.komputation.cuda.layers.continuation.BaseCudaFixedNumberColumnsContinuation
import com.komputation.instructions.Resourceful
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class CudaNormalization internal constructor(
    name : String? = null,
    numberRows : Int,
    numberColumns : Int,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: (Int) -> Kernel,
    private val maximumNumberThreadsPerBlock: Int,
    private val warpSize: Int) : BaseCudaFixedNumberColumnsContinuation(name, numberRows, numberRows, numberColumns), Resourceful {

    private var forwardKernel : Kernel? = null
    private var backwardKernel : Kernel? = null

    private val deviceSums = Pointer()
    private val pointerToDeviceSums = Pointer.to(this.deviceSums)

    private var numberBlocksInXDimensions = -1
    private var numberBlocksInYDimensions = -1
    private var numberThreads = -1
    private var forwardSharedMemoryBytes = -1
    private var backwardSharedMemoryBytes = -1

    private var numberIterations = intArrayOf(-1)
    private val pointerToNumberIterations = Pointer.to(this.numberIterations)

    override fun acquire(maximumBatchSize : Int) {
        super.acquire(maximumBatchSize)

        val numberBatchColumns = maximumBatchSize * this.maximumInputColumns
        allocateDeviceFloatMemory(this.deviceSums, numberBatchColumns)

        val launchConfiguration = computeColumnwiseLaunchConfiguration(this.numberInputRows, this.maximumInputColumns, this.maximumNumberThreadsPerBlock)

        this.numberBlocksInXDimensions = maximumBatchSize
        this.numberBlocksInYDimensions = launchConfiguration.numberBlocks
        this.numberThreads = launchConfiguration.numberThreadsPerBlock
        this.numberIterations[0] = launchConfiguration.numberIterations

        val numberWarps = (this.numberInputRows / launchConfiguration.numberIterations + this.warpSize - 1) / this.warpSize

        this.forwardSharedMemoryBytes = computeDeviceFloatArraySize(numberWarps).toInt()
        this.backwardSharedMemoryBytes = computeDeviceFloatArraySize(numberWarps).toInt()

        this.forwardKernel = this.createForwardKernel()
        this.backwardKernel = this.createBackwardKernel(this.numberThreads)
    }

    override fun release() {
        super.release()

        this.backwardKernel!!.destroy()

        cudaFree(this.deviceSums)
        this.forwardKernel!!.destroy()
    }

    override fun computeForwardResult(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, batchMaximumInputLength: Int, isTraining: Boolean) {
        val parameters = Pointer.to(
            this.pointerToBatchSize,
            this.pointerToNumberInputRows,
            this.pointerToMaximumInputEntries,
            this.pointerToNumberIterations,
            Pointer.to(deviceInput),
            this.pointerToDeviceSums,
            this.pointerToForwardResult
        )

        this.forwardKernel!!.launch(
            parameters,
            this.numberBlocksInXDimensions,
            this.numberBlocksInYDimensions,
            this.numberThreads,
            this.forwardSharedMemoryBytes)
    }

    override fun computeBackwardResult(batchSize: Int, chain: Pointer) {
        val parameters = Pointer.to(
            this.pointerToBatchSize,
            this.pointerToNumberInputRows,
            this.pointerToMaximumInputEntries,
            this.pointerToNumberIterations,
            Pointer.to(chain),
            this.pointerToForwardResult,
            this.pointerToDeviceSums,
            this.pointerToBackwardResult
        )

        this.backwardKernel!!.launch(
            parameters,
            this.numberBlocksInXDimensions,
            this.numberBlocksInYDimensions,
            this.numberThreads,
            this.backwardSharedMemoryBytes)
    }

}