package shape.komputation.cuda.loss

import jcuda.Pointer
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

        this.forwardPtxFile = acquireKernel(File(javaClass.getResource("/cuda/SquaredLoss.cu").toURI()), "squaredLossKernel", this.forwardKernel)
        this.backwardPtxFile = acquireKernel(File(javaClass.getResource("/cuda/BackwardSquaredLoss.cu").toURI()), "backwardSquaredLossKernel", this.backwardKernel)

        allocateDeviceMemory(this.deviceForwardResults, this.targetDimension)
        allocateDeviceMemory(this.deviceLoss, 1)
        allocateDeviceMemory(this.deviceBackwardResults, this.targetDimension)

    }

    private fun acquireKernel(cuFile : File, kernelName: String, kernel: CUfunction): File {

        val ptxFile = File.createTempFile(kernelName, ".ptx")
        ptxFile.deleteOnExit()

        val ptxPath = ptxFile.path

        compileKernel(cuFile.path, ptxPath, this.capabilities)

        loadKernel(ptxPath, kernel, kernelName)

        return ptxFile

    }

    override fun release() {

        this.forwardPtxFile!!.delete()
        this.backwardPtxFile!!.delete()

        cudaFree(this.deviceForwardResults)
        cudaFree(this.deviceLoss)

        cudaFree(this.deviceBackwardResults)

    }

    override fun accumulate(predictions: Pointer, targets: Pointer) {

        val parameters = Pointer.to(
            this.deviceTargetDimension,
            Pointer.to(predictions),
            Pointer.to(targets),
            this.pointerToDeviceForwardResults,
            this.pointerToDeviceSum)

        launchKernel(this.forwardKernel, parameters, 1, this.targetDimension, this.targetDimension)

    }

    override fun accessAccumulation() =

        getVector(this.deviceLoss, 1)[0]

    override fun reset() {

        setVectorToZero(this.deviceLoss, 1)

    }

    override fun backward(predictions: Pointer, targets: Pointer): Pointer {

        val parameters = Pointer.to(
            this.deviceTargetDimension,
            Pointer.to(predictions),
            Pointer.to(targets),
            this.pointerToDeviceBackwardResults)

        launchKernel(this.backwardKernel, parameters, 1, this.targetDimension)

        return this.deviceBackwardResults

    }

}