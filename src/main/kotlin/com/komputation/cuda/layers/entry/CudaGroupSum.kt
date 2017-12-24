package com.komputation.cuda.layers.entry

import jcuda.Pointer
import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeEntrywiseLaunchConfiguration
import com.komputation.instructions.Resourceful
import jcuda.runtime.JCuda.cudaFree

class CudaGroupSum(
    private val dimension: Int,
    private val parametersPerInstance: Int,
    private val hashTableSize : Int,
    private val createGroupSumKernel : () -> Kernel,
    private val createResetKernel : () -> Kernel,
    private val numberMultiprocessors : Int,
    private val numberResidentWarps: Int,
    private val warpSize: Int,
    private val maximumNumberThreadsPerBlock : Int) : Resourceful {

    private val pointerToDimension = Pointer.to(intArrayOf(this.dimension))
    private val pointerToParametersPerInstance = Pointer.to(intArrayOf(this.parametersPerInstance))

    private var groupSumKernel: Kernel? = null
    private var resetKernel: Kernel? = null

    private val deviceGroupSum = Pointer()
    fun getDeviceSum() = this.deviceGroupSum
    private val pointerToGroupSum = Pointer.to(this.deviceGroupSum)
    fun getPointerToSum() = this.pointerToGroupSum

    private val pointerToZero = Pointer.to(floatArrayOf(0f))

    private var maximumBatchSize = -1
    private var groupSumSize = intArrayOf(-1)
    private val pointerToGroupSumSize = Pointer.to(this.groupSumSize)

    private val reset_numberIterations = intArrayOf(-1)
    private val reset_pointerToNumberIterations = Pointer.to(this.reset_numberIterations)
    private var reset_numberBlocks = -1
    private var reset_numberThreadsPerBlock = -1

    override fun acquire(maximumBatchSize: Int) {
        this.maximumBatchSize = maximumBatchSize

        this.groupSumSize[0] = this.maximumBatchSize * this.hashTableSize * this.dimension
        allocateDeviceFloatMemory(this.deviceGroupSum, this.groupSumSize[0])

        this.groupSumKernel = this.createGroupSumKernel()
        this.resetKernel = this.createResetKernel()

        val resetConfiguration = computeEntrywiseLaunchConfiguration(this.groupSumSize[0], this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreadsPerBlock)
        this.reset_numberIterations[0] = resetConfiguration.numberIterations
        this.reset_numberBlocks = resetConfiguration.numberBlocks
        this.reset_numberThreadsPerBlock = resetConfiguration.numberThreadsPerBlock
    }

    fun reset() {
        this.resetKernel!!.launch(
            Pointer.to(
                this.pointerToGroupSumSize,
                this.reset_pointerToNumberIterations,
                this.pointerToGroupSum,
                this.pointerToZero
            ),
            this.reset_numberBlocks,
            1,
            this.reset_numberThreadsPerBlock,
            0)
    }

    fun sum(pointerToMapping : Pointer, pointerToInput: Pointer) {
        this.groupSumKernel!!.launch(
            Pointer.to(
                this.pointerToDimension,
                this.pointerToParametersPerInstance,
                pointerToMapping,
                pointerToInput,
                this.pointerToGroupSum
            ),
            this.maximumBatchSize,
            this.parametersPerInstance,
            this.dimension,
            0)
    }

    override fun release() {
        this.maximumBatchSize = -1
        this.groupSumSize[0] = -1

        cudaFree(this.deviceGroupSum)

        this.groupSumKernel!!.destroy()
        this.resetKernel!!.destroy()
    }

}