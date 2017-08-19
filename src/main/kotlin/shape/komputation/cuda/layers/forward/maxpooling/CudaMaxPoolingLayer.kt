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
    numberRows : Int,
    override val maximumInputColumns : Int,
    private val createForwardKernel: (Int) -> Kernel,
    private val createBackwardKernel: (Int) -> Kernel,
    private val maximumNumberThreadsPerBlock: Int,
    private val warpSize : Int) : BaseCudaForwardLayer(name), Resourceful {

    private val maximumNumberEntries = numberRows * this.maximumInputColumns
    private val pointerToMaximumNumberEntries = Pointer.to(intArrayOf(this.maximumNumberEntries))

    private val pointerToNumberRows = Pointer.to(intArrayOf(numberRows))

    private var forwardKernel: Kernel? = null
    override val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)
    override val numberOutputRows = numberRows
    override val maximumOutputColumns = 1

    private var backwardKernel: Kernel? = null
    override val numberInputRows = numberRows

    private val batchSize = intArrayOf(-1)
    private val pointerToBatchSize = Pointer.to(this.batchSize)

    private val deviceMaxIndices = Pointer()
    private val pointerToMaxIndices = Pointer.to(this.deviceMaxIndices)

    private var configuration = computeRowwiseLaunchConfiguration(this.numberInputRows, this.maximumInputColumns, this.maximumNumberThreadsPerBlock)
    private val numberWarps = (this.maximumInputColumns+this.warpSize-1)/this.warpSize
    private val forwardSharedMemoryBytes = computeDeviceIntArraySize(this.numberWarps).toInt()

    private var maximumBatchSize = -1

    override val deviceBackwardResult = Pointer()
    private val pointerToBackwardResult = Pointer.to(this.deviceBackwardResult)

    override fun acquire(maximumBatchSize : Int) {

        this.maximumBatchSize = maximumBatchSize
        this.forwardKernel = this.createForwardKernel(this.configuration.numberThreadsPerBlock)
        this.backwardKernel = this.createBackwardKernel(this.configuration.numberThreadsPerBlock)

        allocateDeviceIntMemory(this.deviceMaxIndices, maximumBatchSize * this.numberInputRows)
        allocateDeviceFloatMemory(this.deviceForwardResult, maximumBatchSize * this.numberInputRows)

        allocateDeviceFloatMemory(this.deviceBackwardResult, maximumBatchSize * this.maximumNumberEntries)

    }

    override fun forward(batchSize: Int, deviceNumberInputColumns: Pointer, deviceInput: Pointer, isTraining: Boolean): Pointer {

        this.batchSize[0] = batchSize

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToMaximumNumberEntries,
                Pointer.to(deviceInput),
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

    override fun backward(batchSize: Int, chain: Pointer) : Pointer {

        this.backwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToMaximumNumberEntries,
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