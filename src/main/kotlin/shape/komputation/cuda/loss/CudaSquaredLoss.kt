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
    private val deviceBackwardResults = Pointer()
    private val deviceSum = Pointer()

    override fun acquire() {

        this.forwardPtxFile = acquireKernel(File(javaClass.getResource("/cuda/SquaredLoss.cu").toURI()), "squaredLossKernel", this.forwardKernel)
        this.backwardPtxFile = acquireKernel(File(javaClass.getResource("/cuda/BackwardSquaredLoss.cu").toURI()), "backwardSquaredLossKernel", this.backwardKernel)

        allocateDeviceMemory(this.deviceForwardResults, this.targetDimension)
        allocateDeviceMemory(this.deviceSum, 1)
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
        cudaFree(this.deviceSum)

        cudaFree(this.deviceBackwardResults)

    }

    override fun forward(predictions: Pointer, targets: Pointer): Double {

        val parameters = Pointer.to(
            Pointer.to(intArrayOf(this.targetDimension)),
            Pointer.to(predictions),
            Pointer.to(targets),
            Pointer.to(this.deviceForwardResults),
            Pointer.to(this.deviceSum))

        launchKernel(this.forwardKernel, parameters, 1, this.targetDimension, this.targetDimension)

        val hostResult = getVector(this.deviceSum, 1)[0]

        return hostResult

    }

    override fun backward(predictions: Pointer, targets: Pointer): Pointer {

        val parameters = Pointer.to(
            Pointer.to(intArrayOf(this.targetDimension)),
            Pointer.to(predictions),
            Pointer.to(targets),
            Pointer.to(this.deviceBackwardResults))

        launchKernel(this.backwardKernel, parameters, 1, this.targetDimension)

        return this.deviceBackwardResults

    }

}