package com.komputation.cuda.layers.continuation.dropout

import com.komputation.cpu.functions.seed
import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeNumberOfThreadsForRows
import com.komputation.cuda.layers.continuation.BaseCudaFixedNumberColumnsContinuation
import com.komputation.cuda.setIntArray
import com.komputation.instructions.Resourceful
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import java.util.*

class CudaDropout internal constructor(
    name : String? = null,
    numberRows: Int,
    numberColumns: Int,
    private val random: Random,
    keepProbability : Float,
    private val createTrainingKernel: () -> Kernel,
    private val createRuntimeKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val warpSize : Int,
    private val maximumNumberThreadsPerBlock : Int) : BaseCudaFixedNumberColumnsContinuation(name, numberRows, numberRows, numberColumns), Resourceful {

    private var numberBlocksInXDimension = -1
    private var numberBlocksInYDimension = -1
    private var numberThreadsPerBlock = -1
    private var numberIterations = intArrayOf(-1)
    private val pointerToNumberIterations = Pointer.to(this.numberIterations)

    private val pointerToKeepProbability = Pointer.to(floatArrayOf(keepProbability))
    private val pointerToDropoutProbability = Pointer.to(floatArrayOf(1.0f - keepProbability))

    private val deviceSeeds = Pointer()
    private val pointerToDeviceSeeds = Pointer.to(this.deviceSeeds)

    private val deviceMasks = Pointer()
    private val pointerToDeviceMasks = Pointer.to(this.deviceMasks)

    private var trainingKernel : Kernel? = null
    private var runtimeKernel : Kernel? = null
    private var backwardKernel : Kernel? = null

    private val pointerToDeviceForwardResults = Pointer.to(this.deviceForwardResult)
    private val pointerToDeviceBackwardResults = Pointer.to(this.deviceBackwardResult)

    override fun acquire(maximumBatchSize : Int) {
        super.acquire(maximumBatchSize)

        this.trainingKernel = this.createTrainingKernel()
        this.runtimeKernel = this.createRuntimeKernel()

        val seeds = IntArray(this.forwardResultSize)
        seed(this.random, seeds, this.forwardResultSize)
        setIntArray(seeds, seeds.size, this.deviceSeeds)

        allocateDeviceFloatMemory(this.deviceMasks, this.forwardResultSize)

        this.backwardKernel = this.createBackwardKernel()

        this.numberBlocksInXDimension = maximumBatchSize
        val (numberIterations, numberThreadsPerBlock) = computeNumberOfThreadsForRows(this.numberInputRows, this.warpSize, this.maximumNumberThreadsPerBlock)

        this.numberBlocksInYDimension = this.maximumInputColumns
        this.numberIterations[0] = numberIterations
        this.numberThreadsPerBlock = numberThreadsPerBlock
    }

    override fun release() {
        super.release()

        this.trainingKernel!!.destroy()
        this.runtimeKernel!!.destroy()

        cudaFree(this.deviceSeeds)

        cudaFree(this.deviceMasks)

        this.backwardKernel!!.destroy()

        this.numberBlocksInXDimension = -1
        this.numberBlocksInYDimension = -1
        this.numberIterations[0] = -1
        this.numberThreadsPerBlock = -1
    }

    override fun computeForwardResult(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, isTraining: Boolean) {
        val pointerToInput = Pointer.to(deviceInput)

        if(isTraining) {
            this.trainingKernel!!.launch(
                Pointer.to(
                    this.pointerToBatchSize,
                    this.pointerToNumberInputRows,
                    this.pointerToMaximumInputEntries,
                    this.pointerToNumberIterations,
                    this.pointerToDropoutProbability,
                    pointerToInput,
                    this.pointerToDeviceSeeds,
                    this.pointerToDeviceMasks,
                    this.pointerToDeviceForwardResults
                ),
                this.numberBlocksInXDimension,
                this.numberBlocksInYDimension,
                this.numberThreadsPerBlock,
                0
            )
        }
        else {
            this.runtimeKernel!!.launch(
                Pointer.to(
                    this.pointerToBatchSize,
                    this.pointerToNumberInputRows,
                    this.pointerToMaximumInputEntries,
                    this.pointerToNumberIterations,
                    this.pointerToKeepProbability,
                    pointerToInput,
                    this.pointerToDeviceForwardResults
                ),
                this.numberBlocksInXDimension,
                this.numberBlocksInYDimension,
                this.numberThreadsPerBlock,
                0
            )
        }
    }

    override fun computeBackwardResult(batchSize: Int, chain: Pointer) {
        this.backwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToMaximumInputEntries,
                this.pointerToNumberInputRows,
                this.pointerToNumberIterations,
                Pointer.to(chain),
                this.pointerToDeviceMasks,
                this.pointerToDeviceBackwardResults
            ),
            this.numberBlocksInXDimension,
            this.numberBlocksInYDimension,
            this.numberThreadsPerBlock,
            0
        )
    }

}