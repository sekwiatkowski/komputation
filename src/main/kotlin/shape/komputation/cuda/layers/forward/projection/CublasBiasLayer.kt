package shape.komputation.cuda.layers.forward.projection

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.cuda.functions.cublasBackwardProjectionWrtBias
import shape.komputation.cuda.kernels.Kernel
import shape.komputation.cuda.layers.BaseCudaForwardLayer
import shape.komputation.cuda.optimization.CudaUpdateRule
import shape.komputation.cuda.setFloatArray
import shape.komputation.layers.Resourceful
import shape.komputation.optimization.Optimizable

class CublasBiasLayer internal constructor(
    name: String?,
    private val cublasHandle: cublasHandle,
    private val maximumNumberThreadsPerBlock: Int,
    private val numberInputRows: Int,
    private val numberInputColumns: Int,
    private val createKernel: () -> Kernel,
    private val initialBias: FloatArray,
    private val biasUpdateRule: CudaUpdateRule? = null) : BaseCudaForwardLayer(name), Optimizable, Resourceful {

    private val numberInputEntries = this.numberInputRows * this.numberInputColumns

    private var kernel : Kernel? = null

    private val numberThreadsPerInstance = Math.min(this.numberInputEntries, this.maximumNumberThreadsPerBlock)
    private val numberBlocksPerInstance = (this.numberInputColumns + this.maximumNumberThreadsPerBlock - 1) / this.maximumNumberThreadsPerBlock

    private val deviceForwardResult = Pointer()
    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private val deviceBias = Pointer()
    private val pointerToDeviceBias = Pointer.to(this.deviceBias)

    private val deviceBackwardResult = Pointer()
    private val pointerToDeviceBackwardWrtBias = Pointer.to(this.deviceBackwardResult)

    private val deviceOnes = Pointer()

    private val pointerToNumberEntries = Pointer.to(intArrayOf(this.numberInputEntries))
    private val pointerToNumberInputRows = Pointer.to(intArrayOf(this.numberInputRows))

    private val batchSize = intArrayOf(-1)
    private val pointerToBatchSize = Pointer.to(this.batchSize)

    private var numberBatchInputColumns = -1

    override fun acquire(maximumBatchSize : Int) {

        this.numberBatchInputColumns = maximumBatchSize * this.numberInputColumns

        this.kernel = this.createKernel()

        setFloatArray(this.initialBias, this.numberInputEntries, this.deviceBias)

        val numberBatchResultEntries = maximumBatchSize * this.numberInputEntries
        allocateDeviceFloatMemory(this.deviceForwardResult, numberBatchResultEntries)

        allocateDeviceFloatMemory(this.deviceBackwardResult, this.numberInputEntries)

        this.biasUpdateRule?.acquire(maximumBatchSize)

        setFloatArray(FloatArray(this.numberBatchInputColumns) { 1f }, this.numberBatchInputColumns, this.deviceOnes)

    }

    override fun forward(input : Pointer, batchSize : Int, isTraining : Boolean): Pointer {

        this.batchSize[0] = batchSize

        this.kernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToNumberEntries,
                this.pointerToNumberInputRows,
                Pointer.to(input),
                this.pointerToDeviceBias,
                this.pointerToDeviceForwardResult
            ),
            batchSize,
            this.numberBlocksPerInstance,
            this.numberThreadsPerInstance,
            0
        )

        return this.deviceForwardResult

    }

    override fun backward(chain: Pointer, batchSize : Int): Pointer {

        cublasBackwardProjectionWrtBias(
            this.cublasHandle,
            chain,
            this.numberInputRows,
            this.numberBatchInputColumns,
            this.deviceOnes,
            this.deviceBackwardResult)

        return this.deviceBackwardResult

    }

    override fun optimize(scalingFactor: Float) {

        this.biasUpdateRule?.denseUpdate(this.pointerToDeviceBias, scalingFactor, this.pointerToDeviceBackwardWrtBias)

    }

    override fun release() {

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceBackwardResult)
        cudaFree(this.deviceBias)
        cudaFree(this.deviceOnes)

        this.kernel!!.destroy()

    }

}