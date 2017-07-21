package shape.komputation.cuda.loss

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.Kernel
import shape.komputation.cuda.allocateDeviceMemory
import shape.komputation.cuda.getVector
import shape.komputation.cuda.setVectorToZero

// int length, double *predictions, double *targets, double *result
class CudaLogisticLoss(
    private val forwardKernel : Kernel,
    private val backwardKernel : Kernel,
    private val numberCategories : Int,
    private val numberSteps : Int,
    private val blockSize : Int) : CudaLossFunction {

    private val numberEntries = numberCategories * numberSteps

    private val deviceSums = Pointer()
    private val pointerToDeviceSums = Pointer.to(this.deviceSums)

    private val deviceLoss = Pointer()
    private val pointerToDeviceLoss = Pointer.to(this.deviceLoss)

    private val forwardSharedMemoryBytes = blockSize * Sizeof.DOUBLE

    private val deviceBackwardResult = Pointer()
    private val pointerToBackwardResult = Pointer.to(this.deviceBackwardResult)

    override fun acquire() {

        allocateDeviceMemory(this.deviceSums, this.numberSteps)
        allocateDeviceMemory(this.deviceLoss, 1)

        this.forwardKernel.acquire()

        allocateDeviceMemory(this.deviceBackwardResult, this.numberEntries)

        this.backwardKernel.acquire()

    }

    override fun accumulate(pointerToPredictions: Pointer, pointerToTargets: Pointer) {

        val parameters = Pointer.to(
            pointerToPredictions,
            pointerToTargets,
            this.pointerToDeviceSums,
            this.pointerToDeviceLoss
        )

        this.forwardKernel.launch(
            parameters,
            this.numberSteps,
            this.blockSize,
            this.forwardSharedMemoryBytes)

    }

    override fun accessAccumulation() =

        getVector(this.deviceLoss, 1)[0]

    override fun reset() {

        setVectorToZero(this.deviceLoss, 1)

    }

    override fun backward(pointerToPredictions: Pointer, pointerToTargets: Pointer): Pointer {

        val parameters = Pointer.to(
            pointerToPredictions,
            pointerToTargets,
            this.pointerToBackwardResult
        )

        this.backwardKernel.launch(
            parameters,
            1,
            this.numberEntries,
            0)

        return this.deviceBackwardResult

    }

    override fun release() {

        this.forwardKernel.release()

        cudaFree(this.deviceSums)
        cudaFree(this.deviceLoss)

        this.backwardKernel.release()

        cudaFree(this.deviceBackwardResult)

    }

}