package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.cuda.kernels.Kernel
import shape.komputation.cuda.kernels.launch.computeEntrywiseLaunchConfiguration
import shape.komputation.layers.Resourceful

abstract class BaseCudaEntrywiseActivationLayer internal constructor(
    name: String? = null,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val numberRows: Int,
    private val numberColumns: Int,
    private val numberMultiprocessors : Int,
    private val numberResidentWarps : Int,
    private val warpSize : Int,
    private val maximumNumberThreadsPerBlock : Int) : BaseCudaActivationLayer(name), Resourceful {

    private var numberBlocksInXDimension = -1
    private var numberBlocksInYDimension = -1
    private var numberThreadsPerBlock = -1
    private var numberIterations = intArrayOf(-1)
    private val pointerToNumberIterations = Pointer.to(numberIterations)

    private var forwardKernel : Kernel? = null
    override val deviceNumberOutputColumns = Pointer()
    override val numberOutputRows = this.numberRows
    override val maximumOutputColumns = this.numberColumns
    final override val deviceForwardResult = Pointer()
    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private var backwardKernel : Kernel? = null
    final override val deviceBackwardResult = Pointer()
    override val deviceNumberInputColumns = Pointer()
    override val numberInputRows = this.numberRows
    override val maximumInputColumns = this.numberColumns
    private val pointerToDeviceBackwardResult = Pointer.to(this.deviceBackwardResult)

    private val numberEntries = this.numberRows * this.numberColumns
    private val pointerToNumberEntriesPerInstance = Pointer.to(intArrayOf(this.numberEntries))

    private val batchSize = intArrayOf(-1)
    private val pointerToBatchSize = Pointer.to(this.batchSize)

    override fun acquire(maximumBatchSize: Int) {

        allocateDeviceFloatMemory(this.deviceForwardResult, maximumBatchSize * this.numberEntries)
        this.forwardKernel = this.createForwardKernel()

        allocateDeviceFloatMemory(this.deviceBackwardResult, maximumBatchSize * this.numberEntries)
        this.backwardKernel = this.createBackwardKernel()

        this.numberBlocksInXDimension = maximumBatchSize
        val launchConfiguration = computeEntrywiseLaunchConfiguration(this.numberEntries, this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreadsPerBlock)
        this.numberBlocksInYDimension = launchConfiguration.numberBlocks
        this.numberThreadsPerBlock = launchConfiguration.numberThreadsPerBlock
        this.numberIterations[0] = launchConfiguration.numberIterations

    }

    override fun forward(batchSize: Int, deviceNumberInputColumns: Pointer, deviceInput: Pointer, isTraining: Boolean): Pointer {

        this.batchSize[0] = batchSize

        val forwardParameters = Pointer.to(
            this.pointerToBatchSize,
            this.pointerToNumberEntriesPerInstance,
            this.pointerToNumberIterations,
            Pointer.to(deviceInput),
            this.pointerToDeviceForwardResult
        )

        this.forwardKernel!!.launch(
            forwardParameters,
            this.numberBlocksInXDimension,
            this.numberBlocksInYDimension,
            this.numberThreadsPerBlock,
            0)

        return this.deviceForwardResult

    }

    override fun backward(batchSize: Int, chain: Pointer) : Pointer {

        val backwardParameters = Pointer.to(
            this.pointerToBatchSize,
            this.pointerToNumberEntriesPerInstance,
            this.pointerToNumberIterations,
            this.pointerToDeviceForwardResult,
            Pointer.to(chain),
            this.pointerToDeviceBackwardResult
        )

        this.backwardKernel!!.launch(
            backwardParameters,
            this.numberBlocksInXDimension,
            this.numberBlocksInYDimension,
            this.numberThreadsPerBlock,
            0)

        return this.deviceBackwardResult

    }

    override fun release() {

        this.forwardKernel!!.destroy()
        this.backwardKernel!!.destroy()

        cudaFree(this.deviceForwardResult)

        cudaFree(this.deviceBackwardResult)

        this.numberBlocksInXDimension = -1

    }

}