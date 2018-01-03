package com.komputation.cuda.layers.continuation.projection

import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.functions.cublasBackwardProjectionWrtInput
import com.komputation.cuda.functions.cublasBackwardProjectionWrtWeights
import com.komputation.cuda.functions.cublasMatrixMatrixMultiplication
import com.komputation.cuda.functions.cublasMatrixVectorMultiplication
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeEntrywiseLaunchConfiguration
import com.komputation.cuda.layers.continuation.BaseCudaFixedNumberColumnsContinuation
import com.komputation.cuda.optimization.BaseCudaUpdateRule
import com.komputation.cuda.setArrayToZero
import com.komputation.cuda.setFloatArray
import com.komputation.instructions.Resourceful
import com.komputation.optimization.Optimizable
import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree

class CublasWeighting internal constructor(
    name: String?,
    private val cublasHandle: cublasHandle,
    private val createReplaceNaNKernel: () -> Kernel,
    numberInputRows: Int,
    minimumInputColumns: Int,
    maximumInputColumns: Int,
    numberOutputRows: Int,
    private val initialWeights: FloatArray,
    private val weightUpdateRule: BaseCudaUpdateRule? = null,
    private val numberMultiprocessors : Int,
    private val numberResidentWarps: Int,
    private val warpSize: Int,
    private val maximumNumberThreadsPerBlock : Int) : BaseCudaFixedNumberColumnsContinuation(name, numberInputRows, numberOutputRows, maximumInputColumns), Optimizable, Resourceful {

    private val numberWeightRows = this.numberOutputRows
    private val numberWeightColumns = this.numberInputRows
    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

    private var deviceInput = Pointer()
    private var deviceChainWithoutNaN = Pointer()
    private var pointerToChainWithoutNaN = Pointer.to(this.deviceChainWithoutNaN)
    private var deviceInputWithoutNaN = Pointer()
    private var pointerToInputWithoutNaN = Pointer.to(this.deviceInputWithoutNaN)

    private val deviceWeights = Pointer()
    private val pointerToDeviceWeights = Pointer.to(this.deviceWeights)

    private val deviceBackwardWrtWeights = Pointer()
    private val pointerToDeviceBackwardWrtWeights = Pointer.to(this.deviceBackwardWrtWeights)

    private var replaceNaNKernel : Kernel? = null

    fun getDeviceWeights() =
        this.deviceWeights

    private val canHaveIncompleteLength = minimumInputColumns < maximumInputColumns

    private val replaceChain_numberIterations = intArrayOf(-1)
    private val replaceChain_pointerToNumberIterations = Pointer.to(this.replaceChain_numberIterations)
    private var replaceChain_numberBlocks = -1
    private var replaceChain_numberThreadsPerBlock = -1

    private val replaceInput_numberIterations = intArrayOf(-1)
    private val replaceInput_pointerToNumberIterations = Pointer.to(this.replaceInput_numberIterations)
    private var replaceInput_numberBlocks = -1
    private var replaceInput_numberThreadsPerBlock = -1

    override fun acquire(maximumBatchSize: Int) {
        super.acquire(maximumBatchSize)

        setFloatArray(this.initialWeights, this.numberWeightEntries, this.deviceWeights)
        allocateDeviceFloatMemory(this.deviceBackwardWrtWeights, this.numberWeightEntries)

        this.weightUpdateRule?.acquire(maximumBatchSize)

        if (this.canHaveIncompleteLength) {
            allocateDeviceFloatMemory(this.deviceChainWithoutNaN, this.forwardResultSize)
            allocateDeviceFloatMemory(this.deviceInputWithoutNaN, this.backwardResultSize)

            this.replaceNaNKernel = this.createReplaceNaNKernel()

            val replaceChainConfiguration = computeEntrywiseLaunchConfiguration(this.forwardResultSize, this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreadsPerBlock)
            this.replaceChain_numberIterations[0] = replaceChainConfiguration.numberIterations
            this.replaceChain_numberBlocks = replaceChainConfiguration.numberBlocks
            this.replaceChain_numberThreadsPerBlock = replaceChainConfiguration.numberThreadsPerBlock

            val replaceInputConfiguration = computeEntrywiseLaunchConfiguration(this.backwardResultSize, this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreadsPerBlock)
            this.replaceInput_numberIterations[0] = replaceInputConfiguration.numberIterations
            this.replaceInput_numberBlocks = replaceInputConfiguration.numberBlocks
            this.replaceInput_numberThreadsPerBlock = replaceInputConfiguration.numberThreadsPerBlock
        }
    }

    override fun release() {
        super.release()

        cudaFree(this.deviceWeights)
        cudaFree(this.deviceBackwardWrtWeights)

        if (this.canHaveIncompleteLength) {
            cudaFree(this.deviceChainWithoutNaN)
            cudaFree(this.deviceInputWithoutNaN)

            this.replaceNaNKernel!!.destroy()
        }
    }

    private var pointerToInputLengths = Pointer()

    override fun computeForwardResult(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, batchMaximumInputLength: Int, isTraining: Boolean) {
        this.deviceInput = deviceInput
        this.pointerToInputLengths = Pointer.to(deviceInputLengths)

        if (batchSize == 1 && this.maximumInputColumns == 1) {
            cublasMatrixVectorMultiplication(
                this.cublasHandle,
                this.deviceWeights,
                this.numberWeightRows,
                this.numberWeightColumns,
                this.deviceInput,
                this.deviceForwardResult
            )
        }
        else {
            cublasMatrixMatrixMultiplication(
                this.cublasHandle,
                this.deviceWeights,
                this.numberWeightRows,
                this.numberWeightColumns,
                this.deviceInput,
                this.numberInputRows,
                batchSize * this.maximumInputColumns,
                this.deviceForwardResult)
        }

    }

    override fun computeBackwardResult(batchSize: Int, chain: Pointer) {

        val batchNumberOutputColumns = if (batchSize < this.maximumBatchSize) {
            // Reset the result entries to zero
            setArrayToZero(this.deviceBackwardResult, this.backwardResultSize)

            batchSize * this.maximumOutputColumns
        }
        else {
            this.maximumOutputColumnsInCompleteBatch
        }

        cublasBackwardProjectionWrtInput(
            this.cublasHandle,
            this.deviceWeights,
            this.numberWeightRows,
            this.numberWeightColumns,
            chain,
            this.numberOutputRows,
            batchNumberOutputColumns,
            this.deviceBackwardResult)

        /* chain * input^T
           Suppose the chain is given by.
               3 NaN
               4 NaN
           and the input was:
               1 NaN
               2 NaN
           Then, for chain * input^T
                      1   2
                      NaN NaN
               3 NaN
               4 NaN
           cuBLAS will return:
               NaN NaN
               NaN NaN */

        if(this.canHaveIncompleteLength) {
            this.replaceNaNKernel!!.launch(
                Pointer.to(
                    this.pointerToForwardResultSize,
                    this.pointerToMaximumOutputEntries,
                    this.replaceChain_pointerToNumberIterations,
                    Pointer.to(chain),
                    this.pointerToChainWithoutNaN
                ),
                this.maximumBatchSize,
                this.replaceChain_numberBlocks,
                this.replaceChain_numberThreadsPerBlock,
                0
            )

            this.replaceNaNKernel!!.launch(
                Pointer.to(
                    this.pointerToBackwardResultSize,
                    this.pointerToMaximumInputEntries,
                    this.replaceInput_pointerToNumberIterations,
                    Pointer.to(this.deviceInput),
                    this.pointerToInputWithoutNaN
                ),
                this.maximumBatchSize,
                this.replaceInput_numberBlocks,
                this.replaceInput_numberThreadsPerBlock,
                0
            )

            cublasBackwardProjectionWrtWeights(
                this.cublasHandle,
                this.deviceChainWithoutNaN,
                this.numberOutputRows,
                batchNumberOutputColumns,
                this.deviceInputWithoutNaN,
                this.numberInputRows,
                this.deviceBackwardWrtWeights,
                this.numberWeightEntries)
        }
        else {
            cublasBackwardProjectionWrtWeights(
                this.cublasHandle,
                chain,
                this.numberOutputRows,
                batchNumberOutputColumns,
                this.deviceInput,
                this.numberInputRows,
                this.deviceBackwardWrtWeights,
                this.numberWeightEntries)
        }

    }

    override fun optimize(batchSize: Int) {
        this.weightUpdateRule?.denseUpdate(
            batchSize,
            this.pointerToDeviceWeights,
            this.pointerToDeviceBackwardWrtWeights)

    }

}