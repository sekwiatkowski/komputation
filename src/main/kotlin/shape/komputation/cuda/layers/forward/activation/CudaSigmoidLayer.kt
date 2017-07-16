package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.acquireKernel
import shape.komputation.cuda.allocateDeviceMemory
import shape.komputation.cuda.launchKernel
import shape.komputation.layers.Resourceful
import java.io.File

class CudaSigmoidLayer internal constructor(
    name : String? = null,
    private val computeCapabilities: Pair<Int, Int>,
    maximumThreadsPerBlock: Int,
    private val inputDimension : Int) : BaseCudaActivationLayer(name), Resourceful {

    private val resultDimension = this.inputDimension

    private val numberThreads = Math.min(this.inputDimension, maximumThreadsPerBlock)
    private val numberBlocks = Math.ceil(this.inputDimension.toDouble() / this.numberThreads.toDouble()).toInt()

    private val deviceForwardResult = Pointer()
    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private var forwardPtxFile: File? = null
    private val forwardFunction = CUfunction()

    private val deviceBackwardResult = Pointer()
    private val pointerToDeviceBackwardResult = Pointer.to(this.deviceBackwardResult)

    private var backwardPtxFile: File? = null
    private val backwardFunction = CUfunction()

    val deviceInputDimension = Pointer.to(intArrayOf(this.inputDimension))

    override fun acquire() {

        this.forwardPtxFile = acquireKernel(
            File(this.javaClass.getResource("/cuda/sigmoid/Sigmoid.cu").toURI()),
            "sigmoidKernel",
            this.forwardFunction,
            this.computeCapabilities)

        this.backwardPtxFile = acquireKernel(
            File(this.javaClass.getResource("/cuda/sigmoid/BackwardSigmoid.cu").toURI()),
            "backwardSigmoidKernel",
            this.backwardFunction,
            this.computeCapabilities)

        allocateDeviceMemory(this.deviceForwardResult, this.resultDimension)
        allocateDeviceMemory(this.deviceBackwardResult, this.inputDimension)

    }

    override fun forward(input : Pointer): Pointer {

        val forwardParameters = Pointer.to(
            this.deviceInputDimension,
            Pointer.to(input),
            this.pointerToDeviceForwardResult
        )

        launchKernel(this.forwardFunction, forwardParameters, this.numberBlocks, this.numberThreads, 0)

        return this.deviceForwardResult

    }

    override fun backward(chain : Pointer) : Pointer {

        val backwardParameters = Pointer.to(
            this.deviceInputDimension,
            this.pointerToDeviceForwardResult,
            Pointer.to(chain),
            pointerToDeviceBackwardResult
        )

        launchKernel(this.backwardFunction, backwardParameters, this.numberBlocks, this.numberThreads, 0)

        return this.deviceBackwardResult

    }

    override fun release() {

        this.forwardPtxFile!!.delete()
        this.backwardPtxFile!!.delete()

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceBackwardResult)

    }

}