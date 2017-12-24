package com.komputation.cuda.layers.continuation.projection

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import com.komputation.cuda.*
import com.komputation.cuda.functions.cublasBackwardProjectionWrtBias
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeNumberOfThreadsForRows
import com.komputation.cuda.layers.continuation.BaseCudaFixedNumberColumnsContinuation
import com.komputation.cuda.optimization.BaseCudaUpdateRule
import com.komputation.optimization.Optimizable

class CudaBias internal constructor(
    name: String?,
    private val cublasHandle: cublasHandle,
    numberRows: Int,
    maximumInputColumns: Int,
    private val initialBias: FloatArray,
    private val biasUpdateRule: BaseCudaUpdateRule?,
    private val createKernel: () -> Kernel,
    private val warpSize : Int,
    private val maximumNumberThreadsPerBlock: Int) : BaseCudaFixedNumberColumnsContinuation(name, numberRows, numberRows, maximumInputColumns), Optimizable {

    private var kernel : Kernel? = null

    private var numberBlocksInXDimension = -1
    private var numberBlocksInYDimension = -1
    private var numberThreadsPerBlock = -1

    private val deviceBias = Pointer()
    private val pointerToDeviceBias = Pointer.to(this.deviceBias)

    private val pointerToDeviceBackwardWrtBias = Pointer.to(this.deviceBackwardResult)

    private val deviceOnes = Pointer()

    private val numberIterations = intArrayOf(-1)
    private val pointerToNumberIterations = Pointer.to(numberIterations)

    fun getDeviceBias() =
        this.deviceBias

    override fun acquire(maximumBatchSize : Int) {
        super.acquire(maximumBatchSize)

        this.kernel = this.createKernel()

        setFloatArray(this.initialBias, this.initialBias.size, this.deviceBias)
        setFloatArray(FloatArray(this.maximumBatchInputColumns) { 1f }, this.maximumBatchInputColumns, this.deviceOnes)

        this.biasUpdateRule?.acquire(maximumBatchSize)
        this.numberBlocksInXDimension = maximumBatchSize
        this.numberBlocksInYDimension = this.maximumInputColumns
        val (numberIterations, numberThreadsPerBlock) = computeNumberOfThreadsForRows(this.numberInputRows, this.warpSize, this.maximumNumberThreadsPerBlock)
        this.numberThreadsPerBlock = numberIterations
        this.numberIterations[0] = numberThreadsPerBlock
    }

    override fun release() {
        super.release()

        cudaFree(this.deviceBias)
        cudaFree(this.deviceOnes)

        this.kernel!!.destroy()
    }

    private var pointerToInputLengths = Pointer()

    override fun computeForwardResult(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, batchMaximumInputLength: Int, isTraining: Boolean) {
        this.pointerToInputLengths = Pointer.to(deviceInputLengths)

        this.kernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                Pointer.to(deviceInputLengths),
                this.pointerToMaximumInputEntries,
                this.pointerToNumberInputRows,
                this.pointerToNumberIterations,
                Pointer.to(deviceInput),
                this.pointerToDeviceBias,
                this.pointerToForwardResult
            ),
            this.numberBlocksInXDimension,
            this.numberBlocksInYDimension,
            this.numberThreadsPerBlock,
            0
        )
    }

    override fun computeBackwardResult(batchSize: Int, chain: Pointer) {
        if(batchSize < this.maximumBatchSize) {
            setArrayToZero(this.deviceBackwardResult, this.backwardResultSize[0])

            cublasBackwardProjectionWrtBias(
                this.cublasHandle,
                chain,
                this.numberInputRows,
                batchSize * this.maximumInputColumns,
                this.deviceOnes,
                this.deviceBackwardResult)
        }
        else {
            cublasBackwardProjectionWrtBias(
                this.cublasHandle,
                chain,
                this.numberOutputRows,
                this.maximumBatchInputColumns,
                this.deviceOnes,
                this.deviceBackwardResult)
        }
    }

    override fun optimize(batchSize: Int) {
        this.biasUpdateRule?.denseUpdate(
            batchSize,
            this.pointerToDeviceBias,
            this.pointerToDeviceBackwardWrtBias)
    }

}