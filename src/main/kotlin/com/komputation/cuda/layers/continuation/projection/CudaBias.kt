package com.komputation.cuda.layers.continuation.projection

import com.komputation.cuda.computeDeviceFloatArraySize
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeNumberOfThreadsForRows
import com.komputation.cuda.layers.continuation.BaseCudaFixedNumberColumnsContinuation
import com.komputation.cuda.optimization.BaseCudaUpdateRule
import com.komputation.cuda.setFloatArray
import com.komputation.optimization.Optimizable
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class CudaBias internal constructor(
    name: String?,
    numberRows: Int,
    maximumInputColumns: Int,
    private val initialBias: FloatArray,
    private val biasUpdateRule: BaseCudaUpdateRule?,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val warpSize : Int,
    private val maximumNumberThreadsPerBlock: Int) : BaseCudaFixedNumberColumnsContinuation(name, numberRows, numberRows, maximumInputColumns), Optimizable {

    private var forwardKernel : Kernel? = null
    private var forward_numberBlocksInXDimension = -1
    private var forward_numberBlocksInYDimension = -1
    private var forward_numberThreadsPerBlock = -1
    private val forward_numberIterations = intArrayOf(-1)
    private val forward_pointerToNumberIterations = Pointer.to(this.forward_numberIterations)

    private val backward_numberIterations = intArrayOf(-1)
    private val backward_pointerToNumberIterations = Pointer.to(this.backward_numberIterations)

    private var backwardKernel : Kernel? = null

    private val deviceBias = Pointer()
    private val pointerToDeviceBias = Pointer.to(this.deviceBias)

    private val pointerToDeviceBackwardWrtBias = Pointer.to(this.deviceBackwardResult)

    fun getDeviceBias() =
        this.deviceBias

    override fun acquire(maximumBatchSize : Int) {
        super.acquire(maximumBatchSize)

        this.forwardKernel = this.createForwardKernel()
        this.backwardKernel = this.createBackwardKernel()

        setFloatArray(this.initialBias, this.initialBias.size, this.deviceBias)

        this.forward_numberBlocksInXDimension = maximumBatchSize
        this.forward_numberBlocksInYDimension = this.maximumInputColumns
        val (forward_numberIterations, forward_numberThreadsPerBlock) = computeNumberOfThreadsForRows(this.numberInputRows, this.warpSize, this.maximumNumberThreadsPerBlock)
        this.forward_numberThreadsPerBlock = forward_numberThreadsPerBlock
        this.forward_numberIterations[0] = forward_numberIterations

        this.biasUpdateRule?.acquire(maximumBatchSize)
    }

    override fun release() {
        super.release()

        cudaFree(this.deviceBias)

        this.forwardKernel!!.destroy()
        this.backwardKernel!!.destroy()

        this.forward_numberBlocksInXDimension = -1
        this.forward_numberBlocksInYDimension = -1
        this.forward_numberThreadsPerBlock = -1
        this.forward_numberIterations[0] = -1

        this.backward_numberIterations[0] = -1
    }

    private var pointerToInputLengths = Pointer()

    private var isTraining = false
    override fun computeForwardResult(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, batchMaximumInputLength: Int, isTraining: Boolean) {
        this.isTraining = isTraining
        this.pointerToInputLengths = Pointer.to(deviceInputLengths)

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                Pointer.to(deviceInputLengths),
                this.pointerToMaximumInputEntries,
                this.pointerToNumberInputRows,
                this.forward_pointerToNumberIterations,
                Pointer.to(deviceInput),
                this.pointerToDeviceBias,
                this.pointerToForwardResult
            ),
            this.forward_numberBlocksInXDimension,
            this.forward_numberBlocksInYDimension,
            this.forward_numberThreadsPerBlock,
            0
        )

    }

    private val maximumOutputColumnsInCurrentBatchArray = intArrayOf(-1)
    private val pointerToMaximumOutputColumnsInCurrentBatch = Pointer.to(maximumOutputColumnsInCurrentBatchArray)

    override fun computeBackwardResult(batchSize: Int, chain: Pointer) {
        this.maximumOutputColumnsInCurrentBatchArray[0] = batchSize * this.maximumOutputColumns

        val (numberIterations, numberThreads, numberWarps) = computeNumberOfThreadsForRows(this.maximumOutputColumnsInCurrentBatchArray[0], this.warpSize, this.maximumNumberThreadsPerBlock)
        this.backward_numberIterations[0] = numberIterations

        this.backwardKernel!!.launch(
            Pointer.to(
                Pointer.to(chain),
                this.pointerToNumberOutputRows,
                this.pointerToMaximumOutputColumnsInCurrentBatch,
                this.backward_pointerToNumberIterations,
                this.pointerToDeviceBackwardWrtBias
            ),
            this.numberOutputRows,
            1,
            numberThreads,
            computeDeviceFloatArraySize(numberWarps).toInt()
        )

    }

    override fun optimize(batchSize: Int) {
        this.biasUpdateRule?.denseUpdate(
            batchSize,
            this.pointerToDeviceBias,
            this.pointerToDeviceBackwardWrtBias)
    }

}