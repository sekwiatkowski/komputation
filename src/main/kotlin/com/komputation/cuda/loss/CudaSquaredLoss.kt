package com.komputation.cuda.loss

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.computeDeviceFloatArraySize
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import com.komputation.cuda.kernels.launch.computeEntrywiseLaunchConfiguration

class CudaSquaredLoss internal constructor(
    private val targetsPerStep : Int,
    private val numberSteps: Int,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val numberMultiprocessors : Int,
    private val numberResidentWarps : Int,
    private val warpSize : Int,
    private val maximumNumberThreadsPerBlock : Int) : CudaLossFunction {

    private val pointerToTargetsPerStep = Pointer.to(intArrayOf(this.targetsPerStep))

    private val targetsPerInstance = this.numberSteps * this.targetsPerStep
    private val pointerToTargetsPerInstance = Pointer.to(intArrayOf(this.targetsPerInstance))

    private val batchSize = intArrayOf(-1)
    private val pointerToBatchSize = Pointer.to(this.batchSize)
    private var maximumBatchSize = -1
    private var maximumTargets = -1

    private val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)

    private val deviceBackwardResults = Pointer()
    private val pointerToBackwardResults = Pointer.to(this.deviceBackwardResults)

    private var forwardKernel : Kernel? = null
    private var forwardNumberBlocks = -1
    private var forwardNumberThreadsPerBlock = -1
    private var forwardNumberIterations = intArrayOf(-1)
    private var pointerToForwardNumberIterations = Pointer.to(this.forwardNumberIterations)
    private var forwardSharedMemoryBytes = -1

    private var backwardKernel : Kernel? = null
    private var backwardNumberBlocks = -1
    private var backwardNumberIterations = intArrayOf(-1)
    private var pointerToBackwardNumberIterations = Pointer.to(this.backwardNumberIterations)
    private var backwardNumberThreadsPerBlock = -1

    override fun acquire(maximumBatchSize: Int) {
        this.maximumBatchSize = maximumBatchSize
        this.maximumTargets = this.maximumBatchSize * this.targetsPerInstance

        allocateDeviceFloatMemory(this.deviceForwardResult, this.maximumBatchSize)
        allocateDeviceFloatMemory(this.deviceBackwardResults, this.maximumTargets)

        val forwardLaunchConfiguration = computeColumnwiseLaunchConfiguration(this.targetsPerStep, this.numberSteps, this.maximumNumberThreadsPerBlock)
        this.forwardNumberBlocks = forwardLaunchConfiguration.numberBlocks
        this.forwardNumberThreadsPerBlock = forwardLaunchConfiguration.numberThreadsPerBlock
        this.forwardNumberIterations[0] = forwardLaunchConfiguration.numberIterations
        val numberForwardWarps = (this.targetsPerStep / forwardLaunchConfiguration.numberIterations + this.warpSize - 1) / this.warpSize
        this.forwardSharedMemoryBytes =  computeDeviceFloatArraySize(numberForwardWarps).toInt()

        this.forwardKernel = this.createForwardKernel()

        val backwardLaunchConfiguration = computeEntrywiseLaunchConfiguration(this.targetsPerInstance, this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreadsPerBlock)
        this.backwardNumberBlocks = backwardLaunchConfiguration.numberBlocks
        this.backwardNumberThreadsPerBlock = backwardLaunchConfiguration.numberThreadsPerBlock
        this.backwardNumberIterations[0] = backwardLaunchConfiguration.numberIterations

        this.backwardKernel = this.createBackwardKernel()
    }

    override fun release() {
        cudaFree(this.deviceBackwardResults)

        this.backwardKernel!!.destroy()

        cudaFree(this.deviceForwardResult)

        this.forwardKernel!!.destroy()

        this.maximumBatchSize = -1
    }

    // int batchSize, int numberRows, int numberEntriesPerInstance, int numberIterations
    override fun accumulate(pointerToPredictions: Pointer, pointerToTargets: Pointer, batchSize : Int) {
        this.batchSize[0] = batchSize

        val parameters = Pointer.to(
            this.pointerToBatchSize,
            this.pointerToTargetsPerStep,
            this.pointerToTargetsPerInstance,
            this.pointerToForwardNumberIterations,
            pointerToPredictions,
            pointerToTargets,
            this.pointerToForwardResult)

        this.forwardKernel!!.launch(
            parameters,
            this.maximumBatchSize,
            this.forwardNumberBlocks,
            this.forwardNumberThreadsPerBlock,
            this.forwardSharedMemoryBytes)
    }

    override fun accessAccumulation(): Float {
        val sums = getFloatArray(this.deviceForwardResult, this.maximumBatchSize)

        var loss = 0.0f
        for(sum in sums) {
            loss += sum
        }

        return loss
    }

    override fun backward(batchSize: Int, pointerToPredictions: Pointer, pointerToTargets: Pointer): Pointer {
        this.accessAccumulation()

        val parameters = Pointer.to(
            this.pointerToBatchSize,
            this.pointerToTargetsPerInstance,
            this.pointerToBackwardNumberIterations,
            pointerToPredictions,
            pointerToTargets,
            this.pointerToBackwardResults)

        this.backwardKernel!!.launch(
            parameters,
            this.maximumBatchSize,
            this.backwardNumberBlocks,
            this.backwardNumberThreadsPerBlock,
            0)

        return this.deviceBackwardResults
    }

}