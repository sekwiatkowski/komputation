package shape.komputation.cuda.layers.forward.projection

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.cuda.functions.cublasBackwardProjectionWrtBias
import shape.komputation.cuda.kernels.Kernel
import shape.komputation.cuda.kernels.launch.computeEntrywiseLaunchConfiguration
import shape.komputation.cuda.layers.BaseCudaForwardLayer
import shape.komputation.cuda.optimization.CudaUpdateRule
import shape.komputation.cuda.setFloatArray
import shape.komputation.layers.Resourceful
import shape.komputation.optimization.Optimizable

class CublasBiasLayer internal constructor(
    name: String?,
    private val cublasHandle: cublasHandle,
    override val numberInputRows: Int,
    override val numberInputColumns: Int,
    private val initialBias: FloatArray,
    private val biasUpdateRule: CudaUpdateRule?,
    private val createKernel: () -> Kernel,
    private val numberMultiprocessors : Int,
    private val numberResidentWarps : Int,
    private val warpSize : Int,
    private val maximumNumberThreadsPerBlock: Int) : BaseCudaForwardLayer(name), Optimizable, Resourceful {

    private val numberInputEntries = this.numberInputRows * this.numberInputColumns

    private var kernel : Kernel? = null

    private var numberBlocks = -1
    private var numberThreadsPerBlock = -1

    override val deviceForwardResult = Pointer()
    override val numberOutputRows = this.numberInputRows
    override val numberOutputColumns = this.numberInputColumns
    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private val deviceBias = Pointer()
    private val pointerToDeviceBias = Pointer.to(this.deviceBias)

    override val deviceBackwardResult = Pointer()
    private val pointerToDeviceBackwardWrtBias = Pointer.to(this.deviceBackwardResult)

    private val deviceOnes = Pointer()

    private val pointerToNumberEntries = Pointer.to(intArrayOf(this.numberInputEntries))
    private val pointerToNumberInputRows = Pointer.to(intArrayOf(this.numberInputRows))

    private val batchSize = intArrayOf(-1)
    private val pointerToBatchSize = Pointer.to(this.batchSize)

    private val numberIterations = intArrayOf(-1)
    private val pointerToNumberIterations = Pointer.to(this.numberIterations)

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

        val launchConfiguration = computeEntrywiseLaunchConfiguration(this.numberInputEntries, this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreadsPerBlock)
        this.numberBlocks = launchConfiguration.numberBlocks
        this.numberThreadsPerBlock = launchConfiguration.numberThreadsPerBlock
        this.numberIterations[0] = launchConfiguration.numberIterations

    }

    override fun forward(batchSize: Int, numberInputColumns: Int, input: Pointer, isTraining: Boolean): Pointer {

        this.batchSize[0] = batchSize

        this.kernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToNumberEntries,
                this.pointerToNumberInputRows,
                this.pointerToNumberIterations,
                Pointer.to(input),
                this.pointerToDeviceBias,
                this.pointerToDeviceForwardResult
            ),
            batchSize,
            this.numberBlocks,
            this.numberThreadsPerBlock,
            0
        )

        return this.deviceForwardResult

    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {

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