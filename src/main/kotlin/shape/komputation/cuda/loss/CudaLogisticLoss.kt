package shape.komputation.cuda.loss

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.*

// int length, float *predictions, float *targets, float *result
class CudaLogisticLoss(
    private val createForwardKernel : () -> Kernel,
    private val createBackwardKernel : () -> Kernel,
    private val numberCategories : Int,
    private val numberSteps : Int,
    private val blockSize : Int) : CudaLossFunction {

    private val numberEntries = numberCategories * numberSteps

    private var forwardKernel : Kernel? = null

    private val deviceSums = Pointer()
    private val pointerToDeviceSums = Pointer.to(this.deviceSums)

    private val deviceLoss = Pointer()
    private val pointerToDeviceLoss = Pointer.to(this.deviceLoss)

    private val forwardSharedMemoryBytes = computeDeviceFloatArraySize(this.blockSize).toInt()

    private var backwardKernel : Kernel? = null

    private val deviceBackwardResult = Pointer()
    private val pointerToBackwardResult = Pointer.to(this.deviceBackwardResult)

    override fun acquire(maximumBatchSize : Int) {

        allocateDeviceFloatMemory(this.deviceSums, this.numberSteps)
        allocateDeviceFloatMemory(this.deviceLoss, 1)

        this.forwardKernel = this.createForwardKernel()

        allocateDeviceFloatMemory(this.deviceBackwardResult, this.numberEntries)

        this.backwardKernel = this.createBackwardKernel()

    }

    override fun accumulate(pointerToPredictions: Pointer, pointerToTargets: Pointer, batchSize: Int) {

        val parameters = Pointer.to(
            pointerToPredictions,
            pointerToTargets,
            this.pointerToDeviceSums,
            this.pointerToDeviceLoss
        )

        this.forwardKernel!!.launch(
            parameters,
            this.numberSteps,
            batchSize,
            this.blockSize,
            this.forwardSharedMemoryBytes)

    }

    override fun accessAccumulation() =

        getFloatArray(this.deviceLoss, 1)[0]

    override fun reset() {

        setVectorToZero(this.deviceLoss, 1)

    }

    override fun backward(pointerToPredictions: Pointer, pointerToTargets: Pointer, batchSize : Int): Pointer {

        val parameters = Pointer.to(
            pointerToPredictions,
            pointerToTargets,
            this.pointerToBackwardResult
        )

        this.backwardKernel!!.launch(
            parameters,
            1,
            batchSize,
            this.numberEntries,
            0)

        return this.deviceBackwardResult

    }

    override fun release() {

        this.forwardKernel!!.destroy()

        cudaFree(this.deviceSums)
        cudaFree(this.deviceLoss)

        this.backwardKernel!!.destroy()

        cudaFree(this.deviceBackwardResult)

    }

}