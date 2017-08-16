package shape.komputation.cuda.layers.forward.normalization

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.cuda.computeDeviceFloatArraySize
import shape.komputation.cuda.kernels.Kernel
import shape.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import shape.komputation.cuda.layers.forward.activation.BaseCudaActivationLayer
import shape.komputation.layers.Resourceful

class CudaNormalizationLayer internal constructor(
    name : String? = null,
    private val numberRows : Int,
    private val numberColumns : Int,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: (Int) -> Kernel,
    private val maximumNumberThreadsPerBlock: Int,
    private val warpSize: Int) : BaseCudaActivationLayer(name), Resourceful {

    private val numberEntries = this.numberRows * this.numberColumns

    private var forwardKernel : Kernel? = null
    private val deviceForwardResult = Pointer()
    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private var backwardKernel : Kernel? = null
    private val deviceSums = Pointer()
    private val pointerToDeviceSums = Pointer.to(this.deviceSums)

    private val deviceBackwardResult = Pointer()
    private val pointerToDeviceBackwardResult = Pointer.to(this.deviceBackwardResult)

    private var numberBlocksInXDimensions = -1
    private var numberBlocksInYDimensions = -1
    private var numberThreads = -1
    private var forwardSharedMemoryBytes = -1
    private var backwardSharedMemoryBytes = -1

    private var numberIterations = intArrayOf(-1)
    private val pointerToNumberIterations = Pointer.to(this.numberIterations)

    override fun acquire(maximumBatchSize : Int) {

        val numberBatchEntries = maximumBatchSize * this.numberEntries
        val numberBatchColumns = maximumBatchSize * this.numberColumns

        allocateDeviceFloatMemory(this.deviceForwardResult, numberBatchEntries)
        allocateDeviceFloatMemory(this.deviceSums, numberBatchColumns)

        allocateDeviceFloatMemory(this.deviceBackwardResult, numberBatchEntries)

        val launchConfiguration = computeColumnwiseLaunchConfiguration(this.numberRows, this.numberColumns, this.maximumNumberThreadsPerBlock)

        this.numberBlocksInXDimensions = maximumBatchSize
        this.numberBlocksInYDimensions = launchConfiguration.numberBlocks
        this.numberThreads = launchConfiguration.numberThreadsPerBlock
        this.numberIterations[0] = launchConfiguration.numberIterations

        val numberWarps = (this.numberRows / launchConfiguration.numberIterations + this.warpSize - 1) / this.warpSize

        this.forwardSharedMemoryBytes = computeDeviceFloatArraySize(numberWarps).toInt()
        this.backwardSharedMemoryBytes = computeDeviceFloatArraySize(numberWarps).toInt()

        this.forwardKernel = this.createForwardKernel()
        this.backwardKernel = this.createBackwardKernel(this.numberThreads)

    }

    private val forwardBatchSize = intArrayOf(-1)
    private val pointerToForwardBatchSize = Pointer.to(this.forwardBatchSize)

    private val backwardBatchSize = intArrayOf(-1)
    private val pointerToBackwardBatchSize = Pointer.to(this.backwardBatchSize)

    private val pointerToNumberRows = Pointer.to(intArrayOf(this.numberRows))
    private val pointerToNumberEntries = Pointer.to(intArrayOf(this.numberEntries))

    override fun forward(input : Pointer, batchSize : Int, isTraining : Boolean): Pointer {

        this.forwardBatchSize[0] = batchSize

        val parameters = Pointer.to(
            this.pointerToForwardBatchSize,
            this.pointerToNumberRows,
            this.pointerToNumberEntries,
            this.pointerToNumberIterations,
            Pointer.to(input),
            this.pointerToDeviceSums,
            this.pointerToDeviceForwardResult
        )

        this.forwardKernel!!.launch(
            parameters,
            this.numberBlocksInXDimensions,
            this.numberBlocksInYDimensions,
            this.numberThreads,
            this.forwardSharedMemoryBytes)

        return this.deviceForwardResult

    }

    override fun backward(chain : Pointer, batchSize : Int) : Pointer {

        this.backwardBatchSize[0] = batchSize

        val parameters = Pointer.to(
            this.pointerToBackwardBatchSize,
            this.pointerToNumberRows,
            this.pointerToNumberEntries,
            this.pointerToNumberIterations,
            Pointer.to(chain),
            this.pointerToDeviceForwardResult,
            this.pointerToDeviceSums,
            this.pointerToDeviceBackwardResult
        )

        this.backwardKernel!!.launch(
            parameters,
            this.numberBlocksInXDimensions,
            this.numberBlocksInYDimensions,
            this.numberThreads,
            this.backwardSharedMemoryBytes)

        return this.deviceBackwardResult

    }

    override fun release() {

        cudaFree(this.deviceBackwardResult)

        this.backwardKernel!!.destroy()

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceSums)

        this.forwardKernel!!.destroy()

    }

}