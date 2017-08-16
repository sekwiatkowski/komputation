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
    private val numberEntries: Int,
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
    private val deviceForwardResult = Pointer()
    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private var backwardKernel : Kernel? = null
    private val deviceBackwardResult = Pointer()
    private val pointerToDeviceBackwardResult = Pointer.to(this.deviceBackwardResult)

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

    override fun forward(input : Pointer, batchSize : Int, isTraining : Boolean): Pointer {

        this.batchSize[0] = batchSize

        val forwardParameters = Pointer.to(
            this.pointerToBatchSize,
            this.pointerToNumberEntriesPerInstance,
            this.pointerToNumberIterations,
            Pointer.to(input),
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

    override fun backward(chain : Pointer, batchSize: Int) : Pointer {

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