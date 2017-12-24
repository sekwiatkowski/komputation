package com.komputation.cuda.layers.continuation.projection

import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.functions.cublasBackwardProjectionWrtInput
import com.komputation.cuda.functions.cublasBackwardProjectionWrtWeights
import com.komputation.cuda.functions.cublasMatrixMatrixMultiplication
import com.komputation.cuda.functions.cublasMatrixVectorMultiplication
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
    numberInputRows: Int,
    maximumInputColumns: Int,
    numberOutputRows: Int,
    private val initialWeights: FloatArray,
    private val weightUpdateRule: BaseCudaUpdateRule? = null) : BaseCudaFixedNumberColumnsContinuation(name, numberInputRows, numberOutputRows, maximumInputColumns), Optimizable, Resourceful {

    private val numberWeightRows = this.numberOutputRows
    private val numberWeightColumns = this.numberInputRows
    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

    private var deviceInput = Pointer()

    private val deviceWeights = Pointer()
    private val pointerToDeviceWeights = Pointer.to(this.deviceWeights)

    private val deviceBackwardWrtWeights = Pointer()
    private val pointerToDeviceBackwardWrtWeights = Pointer.to(this.deviceBackwardWrtWeights)

    fun getDeviceWeights() =
        this.deviceWeights

    override fun acquire(maximumBatchSize: Int) {
        super.acquire(maximumBatchSize)

        setFloatArray(this.initialWeights, this.numberWeightEntries, this.deviceWeights)
        allocateDeviceFloatMemory(this.deviceBackwardWrtWeights, this.numberWeightEntries)

        this.weightUpdateRule?.acquire(maximumBatchSize)
    }

    override fun release() {
        super.release()

        cudaFree(this.deviceWeights)
        cudaFree(this.deviceBackwardWrtWeights)
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
        if (batchSize < this.maximumBatchSize) {
            // Reset the result entries to zero
            setArrayToZero(this.deviceBackwardResult, this.backwardResultSize[0])

            cublasBackwardProjectionWrtInput(
                this.cublasHandle,
                this.deviceWeights,
                this.numberWeightRows,
                this.numberWeightColumns,
                chain,
                this.numberOutputRows,
                batchSize * this.maximumOutputColumns,
                this.deviceBackwardResult)

            // Reset the result entries to zero
            setArrayToZero(this.deviceBackwardWrtWeights, this.numberWeightEntries)

            cublasBackwardProjectionWrtWeights(
                this.cublasHandle,
                chain,
                this.numberOutputRows,
                batchSize * this.maximumOutputColumns,
                this.deviceInput,
                this.numberInputRows,
                this.deviceBackwardWrtWeights,
                this.numberWeightEntries)
        }
        else {
            cublasBackwardProjectionWrtInput(
                this.cublasHandle,
                this.deviceWeights,
                this.numberWeightRows,
                this.numberWeightColumns,
                chain,
                this.numberOutputRows,
                this.maximumBatchOutputColumns,
                this.deviceBackwardResult)

            cublasBackwardProjectionWrtWeights(
                this.cublasHandle,
                chain,
                this.numberOutputRows,
                this.maximumBatchOutputColumns,
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