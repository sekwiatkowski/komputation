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
    private val createKernel: (Int) -> Kernel,
    private val maximumNumberThreadsPerBlock: Int,
    private val warpSize : Int) : BaseCudaForwardLayer(name), Resourceful {

    private val numberEntries = this.numberRows * this.maximumColumns
    private val pointerToNumberEntries = Pointer.to(intArrayOf(this.numberEntries))

    private var kernel: Kernel? = null

    private val batchSize = intArrayOf(-1)
    private val pointerToBatchSize = Pointer.to(this.batchSize)

    private val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)

    private val deviceMaxIndices = Pointer()
    private val pointerToMaxIndices = Pointer.to(this.deviceMaxIndices)

    private val forwardConfiguration = computeRowwiseLaunchConfiguration(this.numberRows, this.maximumColumns, this.maximumNumberThreadsPerBlock)
    private val numberWarps = (this.maximumColumns+this.warpSize-1)/this.warpSize
    private val forwardSharedMemoryBytes = computeDeviceIntArraySize(this.numberWarps).toInt()

    private var maximumBatchSize = -1

    override fun acquire(maximumBatchSize : Int) {

        this.maximumBatchSize = maximumBatchSize
        this.kernel = this.createKernel(this.forwardConfiguration.numberThreadsPerBlock)

        allocateDeviceIntMemory(this.deviceMaxIndices, maximumBatchSize * this.numberRows)
        allocateDeviceFloatMemory(this.deviceForwardResult, maximumBatchSize * this.numberRows)

    }

    override fun forward(input : Pointer, batchSize : Int, isTraining : Boolean): Pointer {

        this.batchSize[0] = batchSize

        this.kernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToNumberEntries,
                Pointer.to(input),
                this.pointerToMaxIndices,
                this.pointerToForwardResult
            ),
            this.maximumBatchSize,
            this.forwardConfiguration.numberBlocks,
            this.forwardConfiguration.numberThreadsPerBlock,
            this.forwardSharedMemoryBytes
        )

        return this.deviceForwardResult

    }

    override fun backward(chain : Pointer, batchSize : Int) : Pointer {

        TODO()

    }

    override fun release() {

        this.maximumBatchSize = -1
        this.kernel!!.destroy()

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceMaxIndices)

    }

}