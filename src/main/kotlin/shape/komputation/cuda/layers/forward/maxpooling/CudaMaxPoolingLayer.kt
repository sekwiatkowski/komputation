package shape.komputation.cuda.layers.forward.maxpooling

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.cuda.allocateDeviceIntMemory
import shape.komputation.cuda.computeDeviceIntArraySize
import shape.komputation.cuda.kernels.Kernel
import shape.komputation.cuda.kernels.launch.computeRowwiseLaunchConfiguration
import shape.komputation.cuda.layers.BaseCudaForwardLayer
import shape.komputation.layers.Resourceful

class CudaMaxPoolingLayer internal constructor(
    name : String?,
    private val numberRows : Int,
    private val maximumColumns : Int,
    private val createForwardKernel: (Int) -> Kernel,
    private val createBackwardKernel: (Int) -> Kernel,
    private val maximumNumberThreadsPerBlock: Int,
    private val warpSize : Int) : BaseCudaForwardLayer(name), Resourceful {

    private val numberEntries = this.numberRows * this.maximumColumns
    private val pointerToNumberEntries = Pointer.to(intArrayOf(this.numberEntries))

    private val pointerToNumberRows = Pointer.to(intArrayOf(this.numberRows))

    private var forwardKernel: Kernel? = null
    private var backwardKernel: Kernel? = null

    private val batchSize = intArrayOf(-1)
    private val pointerToBatchSize = Pointer.to(this.batchSize)

    private val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)

    private val deviceMaxIndices = Pointer()
    private val pointerToMaxIndices = Pointer.to(this.deviceMaxIndices)

    private val configuration = computeRowwiseLaunchConfiguration(this.numberRows, this.maximumColumns, this.maximumNumberThreadsPerBlock)
    private val numberWarps = (this.maximumColumns+this.warpSize-1)/this.warpSize
    private val forwardSharedMemoryBytes = computeDeviceIntArraySize(this.numberWarps).toInt()

    private var maximumBatchSize = -1

    private val deviceBackwardResult = Pointer()
    private val pointerToBackwardResult = Pointer.to(this.deviceBackwardResult)

    override fun acquire(maximumBatchSize : Int) {

        this.maximumBatchSize = maximumBatchSize
        this.forwardKernel = this.createForwardKernel(this.configuration.numberThreadsPerBlock)
        this.backwardKernel = this.createBackwardKernel(this.configuration.numberThreadsPerBlock)

        allocateDeviceIntMemory(this.deviceMaxIndices, maximumBatchSize * this.numberRows)
        allocateDeviceFloatMemory(this.deviceForwardResult, maximumBatchSize * this.numberRows)

        allocateDeviceFloatMemory(this.deviceBackwardResult, maximumBatchSize * this.numberEntries)

    }

    override fun forward(input : Pointer, batchSize : Int, isTraining : Boolean): Pointer {

        this.batchSize[0] = batchSize

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToNumberEntries,
                Pointer.to(input),
                this.pointerToMaxIndices,
                this.pointerToForwardResult
            ),
            this.maximumBatchSize,
            this.configuration.numberBlocks,
            this.configuration.numberThreadsPerBlock,
            this.forwardSharedMemoryBytes
        )

        return this.deviceForwardResult

    }

    override fun backward(chain : Pointer, batchSize : Int) : Pointer {

        this.backwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToNumberEntries,
                this.pointerToNumberRows,
                this.pointerToMaxIndices,
                Pointer.to(chain),
                this.pointerToBackwardResult
            ),
            this.maximumBatchSize,
            this.configuration.numberBlocks,
            this.configuration.numberThreadsPerBlock,
            0
        )

        return this.deviceBackwardResult

    }

    override fun release() {

        this.maximumBatchSize = -1
        this.forwardKernel!!.destroy()
        this.backwardKernel!!.destroy()

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceBackwardResult)
        cudaFree(this.deviceMaxIndices)

    }

}