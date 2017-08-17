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
import shape.komputation.cuda.setFloatArray
import shape.komputation.layers.Resourceful
import shape.komputation.optimization.Optimizable

class CublasWeightingLayer internal constructor(
    name: String?,
    private val cublasHandle: cublasHandle,
    override val numberInputRows: Int,
    override val numberInputColumns: Int,
    override val numberOutputRows: Int,
    private val initialWeights: FloatArray,
    private val weightUpdateRule: CudaUpdateRule? = null) : BaseCudaForwardLayer(name), Optimizable, Resourceful {

    private val numberInputEntries = this.numberInputRows * this.numberInputColumns

    private val numberWeightRows = this.numberOutputRows
    private val numberWeightColumns = this.numberInputRows
    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

    override val numberOutputColumns = this.numberInputColumns
    private val numberOutputEntries = this.numberOutputRows * this.numberOutputColumns

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
        this.numberBatchInputColumns = maximumBatchSize * this.numberInputColumns
        this.numberBatchOutputColumns = maximumBatchSize * this.numberOutputColumns

        setFloatArray(this.initialWeights, this.numberWeightEntries, this.deviceWeights)
        allocateDeviceFloatMemory(this.deviceBackwardWrtWeights, this.numberWeightEntries)

        this.weightUpdateRule?.acquire(maximumBatchSize)

        val maxiumumNumberOutputEntries = this.numberOutputEntries * maximumBatchSize
        allocateDeviceFloatMemory(this.deviceForwardResult, maxiumumNumberOutputEntries)

        val maxiumumNumberInputEntries = this.numberInputEntries * maximumBatchSize
        allocateDeviceFloatMemory(this.deviceBackwardResult, maxiumumNumberInputEntries)

    }

    override fun forward(batchSize: Int, numberInputColumns: Int, input: Pointer, isTraining: Boolean): Pointer {

        this.deviceInput = input

        if (batchSize == 1 && this.numberInputColumns == 1) {

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

    override fun backward(batchSize: Int, chain: Pointer): Pointer {

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
            this.numberBatchInputColumns,
            this.deviceBackwardWrtWeights,
            this.numberWeightEntries)

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