package shape.komputation.cuda.loss

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.*

class CudaSquaredLoss(
    private val createForwardKernel: (Int) -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val numberEntries: Int) : CudaLossFunction {

    private fun computeBlockSize(numberEntries : Int, maximumBatchSize : Int) =

        maximumBatchSize * Math.pow(2.0, Math.ceil(Math.log(numberEntries.toDouble()) / Math.log(2.0))).toInt()

    private val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)

    private var forwardKernel : Kernel? = null

    private val pointerToNumberEntries = Pointer.to(intArrayOf(this.numberEntries))

    private var numberBatchEntries = intArrayOf(this.numberEntries)
    private val pointerToNumberBatchEntries = Pointer.to(numberBatchEntries)

    private val accumulationSharedMemoryBytes = computeDeviceFloatArraySize(this.numberEntries).toInt()

    private var backwardKernel : Kernel? = null
    private val deviceBackwardResults = Pointer()
    private val pointerToBackwardResults = Pointer.to(this.deviceBackwardResults)

    private val currentBatchSize = intArrayOf(-1)
    private val pointerToBatchSize = Pointer.to(this.currentBatchSize)

    private var maximumBatchSize = -1
    private var blockSize = -1

    override fun acquire(maximumBatchSize: Int) {

        this.maximumBatchSize = maximumBatchSize
        this.blockSize = computeBlockSize(this.numberEntries, maximumBatchSize)
        this.numberBatchEntries[0] = maximumBatchSize * this.numberEntries

        this.forwardKernel = this.createForwardKernel(this.blockSize)

        allocateDeviceFloatMemory(this.deviceForwardResult, 1)

        this.backwardKernel = this.createBackwardKernel()

        allocateDeviceFloatMemory(this.deviceBackwardResults, this.numberEntries * maximumBatchSize)

    }

    override fun release() {

        cudaFree(this.deviceBackwardResults)

        this.backwardKernel!!.destroy()

        cudaFree(this.deviceForwardResult)

        this.forwardKernel!!.destroy()

        this.maximumBatchSize = -1

    }

    override fun accumulate(pointerToPredictions: Pointer, pointerToTargets: Pointer, batchSize : Int) {

        val parameters = Pointer.to(
            this.pointerToNumberBatchEntries,
            pointerToPredictions,
            pointerToTargets,
            this.pointerToForwardResult)

        this.forwardKernel!!.launch(
            parameters,
            1,
            1,
            this.blockSize,
            this.accumulationSharedMemoryBytes)

    }

    override fun accessAccumulation() =

        getFloatArray(this.deviceForwardResult, 1)[0]

    override fun reset() {

        setVectorToZero(this.deviceForwardResult, 1)

    }

    override fun backward(pointerToPredictions: Pointer, pointerToTargets: Pointer, batchSize: Int): Pointer {

        this.currentBatchSize[0] = batchSize

        val parameters = Pointer.to(
            this.pointerToBatchSize,
            this.pointerToNumberEntries,
            pointerToPredictions,
            pointerToTargets,
            this.pointerToBackwardResults)

        this.backwardKernel!!.launch(
            parameters,
            this.maximumBatchSize,
            1,
            this.numberEntries,
            0)

        return this.deviceBackwardResults

    }

}