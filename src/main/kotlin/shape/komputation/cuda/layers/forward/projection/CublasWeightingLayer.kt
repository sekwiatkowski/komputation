package shape.komputation.cuda.layers.forward.projection

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.cuda.functions.cublasBackwardProjectionWrtInput
import shape.komputation.cuda.functions.cublasBackwardProjectionWrtWeights
import shape.komputation.cuda.functions.cublasMatrixMatrixMultiplication
import shape.komputation.cuda.functions.cublasMatrixVectorMultiplication
import shape.komputation.cuda.layers.BaseCudaForwardLayer
import shape.komputation.cuda.optimization.CudaUpdateRule
import shape.komputation.cuda.setArrayToZero
import shape.komputation.cuda.setFloatArray
import shape.komputation.layers.Resourceful
import shape.komputation.optimization.Optimizable

class CublasWeightingLayer internal constructor(
    name: String?,
    private val cublasHandle: cublasHandle,
    override val numberInputRows: Int,
    override val maximumInputColumns: Int,
    override val numberOutputRows: Int,
    private val initialWeights: FloatArray,
    private val weightUpdateRule: CudaUpdateRule? = null) : BaseCudaForwardLayer(name), Optimizable, Resourceful {

    private val numberInputEntries = this.numberInputRows * this.maximumInputColumns

    private val numberWeightRows = this.numberOutputRows
    private val numberWeightColumns = this.numberInputRows
    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

    override val maximumOutputColumns = this.maximumInputColumns
    private val numberOutputEntries = this.numberOutputRows * this.maximumOutputColumns

    private var deviceInput = Pointer()

    private val deviceWeights = Pointer()
    private val pointerToDeviceWeights = Pointer.to(this.deviceWeights)

    override val deviceForwardResult = Pointer()

    private val deviceBackwardWrtWeights = Pointer()
    private val pointerToDeviceBackwardWrtWeights = Pointer.to(this.deviceBackwardWrtWeights)

    override val deviceBackwardResult = Pointer()

    private var maximumBatchSize = -1
    private var numberBatchInputColumns = -1
    private var numberBatchOutputColumns = -1

    override fun acquire(maximumBatchSize: Int) {

        this.maximumBatchSize = maximumBatchSize
        this.numberBatchInputColumns = maximumBatchSize * this.maximumInputColumns
        this.numberBatchOutputColumns = maximumBatchSize * this.maximumOutputColumns

        setFloatArray(this.initialWeights, this.numberWeightEntries, this.deviceWeights)
        allocateDeviceFloatMemory(this.deviceBackwardWrtWeights, this.numberWeightEntries)

        this.weightUpdateRule?.acquire(maximumBatchSize)

        allocateDeviceFloatMemory(this.deviceForwardResult, maximumBatchSize * this.numberOutputEntries)
        allocateDeviceFloatMemory(this.deviceBackwardResult, maximumBatchSize * this.numberInputEntries)

    }

    override fun forward(batchSize: Int, deviceInput: Pointer, isTraining: Boolean): Pointer {

        this.deviceInput = deviceInput

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
                this.numberBatchInputColumns,
                this.deviceForwardResult)

        }

        return this.deviceForwardResult

    }

    private var lastBatchSize = -1

    override fun backward(batchSize: Int, chain: Pointer): Pointer {

        lastBatchSize = batchSize

        if (batchSize < this.maximumBatchSize) {

            // Reset the result entries to zero
            setArrayToZero(this.deviceBackwardResult, this.maximumBatchSize * this.numberInputEntries)

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
                this.numberBatchOutputColumns,
                this.deviceBackwardResult)

            cublasBackwardProjectionWrtWeights(
                this.cublasHandle,
                chain,
                this.numberOutputRows,
                this.numberBatchInputColumns,
                this.deviceInput,
                this.numberInputRows,
                this.deviceBackwardWrtWeights,
                this.numberWeightEntries)

        }

        return this.deviceBackwardResult

    }

    override fun optimize(scalingFactor: Float) {

        this.weightUpdateRule?.denseUpdate(this.pointerToDeviceWeights, scalingFactor, this.pointerToDeviceBackwardWrtWeights)

    }

    override fun release() {

        cudaFree(this.deviceWeights)
        cudaFree(this.deviceForwardResult)

        cudaFree(this.deviceBackwardWrtWeights)
        cudaFree(this.deviceBackwardResult)

        this.maximumBatchSize = -1
        this.numberBatchInputColumns = -1

    }

}