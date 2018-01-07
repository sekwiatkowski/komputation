package com.komputation.cuda.layers.continuation.activation

import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeNumberOfThreadsForRows
import com.komputation.cuda.layers.continuation.BaseCudaFixedNumberColumnsContinuation
import com.komputation.cuda.layers.continuation.CudaActivation
import com.komputation.instructions.Resourceful
import jcuda.Pointer

abstract class BaseCudaEntrywise internal constructor(
    name: String? = null,
    private val numberRows: Int,
    private val numberColumns: Int,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val warpSize: Int,
    private val maximumNumberThreadsPerBlock: Int) : BaseCudaFixedNumberColumnsContinuation(name, numberRows, numberRows, numberColumns), CudaActivation, Resourceful {

    private var numberBlocksInXDimension = -1
    private var numberBlocksInYDimension = -1
    private var numberThreadsPerBlock = -1
    private var numberIterations = intArrayOf(-1)
    private val pointerToNumberIterations = Pointer.to(this.numberIterations)

    private var forwardKernel : Kernel? = null
    private var backwardKernel : Kernel? = null

    override fun acquire(maximumBatchSize: Int) {
        super.acquire(maximumBatchSize)

        this.forwardKernel = this.createForwardKernel()
        this.backwardKernel = this.createBackwardKernel()

        this.numberBlocksInXDimension = maximumBatchSize
        this.numberBlocksInYDimension = this.numberColumns

        val (numberIterations, numberThreadsPerBlock) = computeNumberOfThreadsForRows(this.numberRows, this.warpSize, this.maximumNumberThreadsPerBlock)
        this.numberThreadsPerBlock = numberThreadsPerBlock
        this.numberIterations[0] = numberIterations
    }

    override fun release() {
        super.release()

        this.forwardKernel!!.destroy()
        this.backwardKernel!!.destroy()

        this.numberBlocksInXDimension = -1
    }

    override fun computeForwardResult(batchSize: Int, deviceInput: Pointer, deviceInputLengths : Pointer, isTraining: Boolean) {
        val forwardParameters = Pointer.to(
            this.pointerToBatchSize,
            this.pointerToNumberInputRows,
            this.pointerToMaximumInputEntries,
            this.pointerToNumberIterations,
            Pointer.to(deviceInput),
            this.pointerToForwardResult
        )

        this.forwardKernel!!.launch(
            forwardParameters,
            this.numberBlocksInXDimension,
            this.numberBlocksInYDimension,
            this.numberThreadsPerBlock,
            0)
    }

    override fun computeBackwardResult(batchSize: Int, chain: Pointer) {
        val backwardParameters = Pointer.to(
            this.pointerToBatchSize,
            this.pointerToNumberInputRows,
            this.pointerToMaximumInputEntries,
            this.pointerToNumberIterations,
            this.pointerToForwardResult,
            Pointer.to(chain),
            this.pointerToBackwardResult
        )

        this.backwardKernel!!.launch(
            backwardParameters,
            this.numberBlocksInXDimension,
            this.numberBlocksInYDimension,
            this.numberThreadsPerBlock,
            0)
    }

}