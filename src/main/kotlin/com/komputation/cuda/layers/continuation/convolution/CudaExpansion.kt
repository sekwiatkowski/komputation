package com.komputation.cuda.layers.continuation.convolution

import com.komputation.cpu.functions.computeNumberFilterColumnPositions
import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.KernelLaunchConfiguration
import com.komputation.cuda.layers.continuation.BaseCudaVariableNumberColumnsContinuation
import com.komputation.instructions.Resourceful
import com.komputation.instructions.continuation.convolution.computeNumberExpandedColumns
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class CudaExpansion internal constructor(
    name : String?,
    numberInputRows : Int,
    maximumInputColumns : Int,
    private val numberFilterRowPositions: Int,
    filterHeight : Int,
    filterWidth : Int,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val warpSize : Int,
    maximumNumberThreads : Int) : BaseCudaVariableNumberColumnsContinuation(name, numberInputRows, filterHeight * filterWidth, maximumInputColumns, { inputLength : Int -> computeNumberExpandedColumns(inputLength, filterWidth, numberFilterRowPositions) }), Resourceful {

    private val pointerToFilterHeight = Pointer.to(intArrayOf(filterHeight))
    private val pointerToFilterWidth = Pointer.to(intArrayOf(filterWidth))

    private val pointerToNumberFilterRowPositions = Pointer.to(intArrayOf(this.numberFilterRowPositions))
    private val numberFilterColumnPositions = computeNumberFilterColumnPositions(this.maximumInputColumns, filterWidth)

    private val maximumConvolutions = this.numberFilterRowPositions * this.numberFilterColumnPositions
    private val pointerToNumberConvolutions = Pointer.to(intArrayOf(this.maximumConvolutions))

    private var forwardKernel : Kernel? = null
    private var backwardKernel : Kernel? = null

    private val maximumNumberWarpsPerBlock = maximumNumberThreads / this.warpSize
    private val pointerToNumberWarpsPerBlock = Pointer.to(intArrayOf(this.maximumNumberWarpsPerBlock))

    private val numberForwardBlocksInYDimension = this.maximumOutputColumns
    private val numberForwardThreads = this.numberOutputRows

    private var numberBackwardBlocksInYDimension = -1
    private var numberBackwardThreads = -1
    private val numberBackwardIterations = intArrayOf(-1)
    private val pointerToNumberBackwardIterations = Pointer.to(this.numberBackwardIterations)

    override val deviceForwardLengths = Pointer()
    private val pointerToForwardLengths = Pointer.to(this.deviceForwardLengths)

    override fun acquire(maximumBatchSize: Int) {
        super.acquire(maximumBatchSize)

        this.forwardKernel = this.createForwardKernel()
        this.backwardKernel = this.createBackwardKernel()

        val backwardLaunch = computeBackwardLaunchConfiguration(this.maximumInputEntries, this.numberOutputRows, this.maximumNumberWarpsPerBlock, this.warpSize)

        this.numberBackwardBlocksInYDimension = backwardLaunch.numberBlocks
        this.numberBackwardThreads = backwardLaunch.numberThreadsPerBlock
        this.numberBackwardIterations[0] = backwardLaunch.numberIterations

        allocateDeviceFloatMemory(this.deviceForwardLengths, maximumBatchSize)
    }

    override fun release() {
        super.release()

        this.forwardKernel!!.destroy()
        this.backwardKernel!!.destroy()

        this.numberBackwardBlocksInYDimension = -1
        this.numberBackwardThreads = -1
        this.numberBackwardIterations[0] = -1

        cudaFree(this.deviceForwardResult)
    }

    private fun computeBackwardLaunchConfiguration(numberInputEntries : Int, filterSize : Int, maximumNumberWarpsPerBlock : Int, warpSize: Int): KernelLaunchConfiguration {
        val numberBackwardBlocksInYDimension =
            if(numberInputEntries <= maximumNumberWarpsPerBlock)
                1
            else
                (numberInputEntries + maximumNumberWarpsPerBlock - 1) / maximumNumberWarpsPerBlock

        val numberWarpsPerBlock = if(numberInputEntries < maximumNumberWarpsPerBlock)
            numberInputEntries
        else
            maximumNumberWarpsPerBlock

        val numberThreads = numberWarpsPerBlock * warpSize

        val numberIterations = (warpSize + filterSize - 1)/ warpSize

        return KernelLaunchConfiguration(numberBackwardBlocksInYDimension, numberThreads, numberIterations)
    }

    private var pointerToInputLengths = Pointer()

    override fun computeForwardResult(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, batchMaximumInputLength: Int, isTraining: Boolean) {
        this.pointerToInputLengths = Pointer.to(deviceInputLengths)

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                Pointer.to(deviceInput),
                this.pointerToInputLengths,
                this.pointerToNumberInputRows,
                this.pointerToMaximumInputEntries,
                this.pointerToNumberFilterRowPositions,
                this.pointerToFilterHeight,
                this.pointerToFilterWidth,
                this.pointerToNumberOutputRows,
                this.pointerToMaximumOutputEntries,
                this.pointerToForwardResult,
                this.pointerToForwardLengths
            ),
            this.maximumBatchSize,
            this.numberForwardBlocksInYDimension,
            this.numberForwardThreads,
            0
        )
    }

    override fun computeBackwardResult(batchSize: Int, chain: Pointer) {
        this.backwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToInputLengths,
                this.pointerToNumberBackwardIterations,
                this.pointerToNumberInputRows,
                this.pointerToMaximumInputEntries,
                this.pointerToNumberWarpsPerBlock,
                this.pointerToFilterHeight,
                this.pointerToFilterWidth,
                this.pointerToNumberOutputRows,
                this.pointerToNumberConvolutions,
                this.pointerToNumberFilterRowPositions,
                Pointer.to(chain),
                this.pointerToBackwardResult
            ),
            this.maximumBatchSize,
            this.numberBackwardBlocksInYDimension,
            this.numberBackwardThreads,
            0
        )
    }

}