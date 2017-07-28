package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.Kernel
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.layers.Resourceful
import shape.komputation.matrix.IntMath

abstract class BaseCudaEntrywiseActivationLayer internal constructor(
    name: String? = null,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    maximumThreadsPerBlock: Int,
    private val numberEntries: Int) : BaseCudaActivationLayer(name), Resourceful {

    private var numberBlocksInXDimension = -1
    private val numberThreads = Math.min(this.numberEntries, maximumThreadsPerBlock)
    private val numberBlocksInYDimension = IntMath.ceil(this.numberEntries.toDouble() / this.numberThreads.toDouble())

    private var forwardKernel : Kernel? = null
    private val deviceForwardResult = Pointer()
    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private var backwardKernel : Kernel? = null
    private val deviceBackwardResult = Pointer()
    private val pointerToDeviceBackwardResult = Pointer.to(this.deviceBackwardResult)

    private val pointerToNumberEntriesPerInstance = Pointer.to(intArrayOf(this.numberEntries))

    private val batchSize = intArrayOf(-1)
    private val pointerToBatchSize = Pointer.to(batchSize)

    override fun acquire(maximumBatchSize: Int) {

        allocateDeviceFloatMemory(this.deviceForwardResult, maximumBatchSize * this.numberEntries)
        this.forwardKernel = this.createForwardKernel()

        allocateDeviceFloatMemory(this.deviceBackwardResult, maximumBatchSize * this.numberEntries)
        this.backwardKernel = this.createBackwardKernel()

        this.numberBlocksInXDimension = maximumBatchSize

    }

    override fun forward(input : Pointer, batchSize : Int, isTraining : Boolean): Pointer {

        this.batchSize[0] = batchSize

        val forwardParameters = Pointer.to(
            this.pointerToBatchSize,
            this.pointerToNumberEntriesPerInstance,
            Pointer.to(input),
            this.pointerToDeviceForwardResult
        )

        this.forwardKernel!!.launch(
            forwardParameters,
            this.numberBlocksInXDimension,
            this.numberBlocksInYDimension,
            this.numberThreads,
            0)

        return this.deviceForwardResult

    }

    override fun backward(chain : Pointer, batchSize: Int) : Pointer {

        val backwardParameters = Pointer.to(
            this.pointerToNumberEntriesPerInstance,
            this.pointerToDeviceForwardResult,
            Pointer.to(chain),
            this.pointerToDeviceBackwardResult
        )

        this.backwardKernel!!.launch(backwardParameters, batchSize, this.numberBlocksInYDimension, this.numberThreads, 0)

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