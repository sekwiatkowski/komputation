package shape.komputation.cuda.loss

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.CUfunction
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.*
import shape.komputation.layers.Resourceful
import java.io.File

class CudaSquaredLoss(private val capabilities : Pair<Int, Int>, private val targetDimension : Int) : CudaLossFunction, Resourceful {

    private var forwardPtxFile: File? = null
    private val forwardKernel = CUfunction()

    private var backwardPtxFile: File? = null
    private val backwardKernel = CUfunction()

    private val deviceForwardResults = Pointer()
    private val pointerToDeviceForwardResults = Pointer.to(this.deviceForwardResults)

    private val deviceBackwardResults = Pointer()
    private val pointerToDeviceBackwardResults = Pointer.to(this.deviceBackwardResults)

    private val deviceLoss = Pointer()
    private val pointerToDeviceSum = Pointer.to(this.deviceLoss)

    private val deviceTargetDimension = Pointer.to(intArrayOf(this.targetDimension))

    override fun acquire() {

        this.forwardPtxFile = acquireKernel(
            File(javaClass.getResource("/cuda/squaredloss/SquaredLossKernel.cu").toURI()),
            "squaredLossKernel",
            this.forwardKernel,
            this.capabilities)

        allocateDeviceMemory(this.deviceForwardResults, this.targetDimension)

        this.backwardPtxFile = acquireKernel(
            File(javaClass.getResource("/cuda/squaredloss/BackwardSquaredLossKernel.cu").toURI()),
            "backwardSquaredLossKernel",
            this.backwardKernel,
            this.capabilities)

        allocateDeviceMemory(this.deviceLoss, 1)
        allocateDeviceMemory(this.deviceBackwardResults, this.targetDimension)

    }

    override fun release() {

        this.forwardPtxFile!!.delete()
        this.backwardPtxFile!!.delete()

        cudaFree(this.deviceForwardResults)
        cudaFree(this.deviceLoss)

        cudaFree(this.deviceBackwardResults)

    }

    private val accumulationSharedMemoryBytes = this.targetDimension * Sizeof.DOUBLE

    override fun accumulate(pointerToPredictions: Pointer, pointerToTargets: Pointer) {

        val parameters = Pointer.to(
            this.deviceTargetDimension,
            pointerToPredictions,
            pointerToTargets,
            this.pointerToDeviceForwardResults,
            this.pointerToDeviceSum)

        launchKernel(
            this.forwardKernel,
            parameters,
            1,
            this.targetDimension,
            this.accumulationSharedMemoryBytes)

    }

    override fun accessAccumulation() =

        getVector(this.deviceLoss, 1)[0]

    override fun reset() {

        setVectorToZero(this.deviceLoss, 1)

    }

    override fun backward(pointerToPredictions: Pointer, pointerToTargets: Pointer): Pointer {

        val parameters = Pointer.to(
            this.deviceTargetDimension,
            pointerToPredictions,
            pointerToTargets,
            this.pointerToDeviceBackwardResults)

        launchKernel(this.backwardKernel, parameters, 1, this.targetDimension, 0)

        return this.deviceBackwardResults

    }

}