package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.Kernel
import shape.komputation.cuda.allocateDeviceMemory
import shape.komputation.layers.Resourceful

class CudaSigmoidLayer internal constructor(
    name : String? = null,
    private val forwardKernel: Kernel,
    private val backwardKernel: Kernel,
    maximumThreadsPerBlock: Int,
    private val inputDimension : Int) : BaseCudaActivationLayer(name), Resourceful {

    private val resultDimension = this.inputDimension

    private val numberThreads = Math.min(this.inputDimension, maximumThreadsPerBlock)
    private val numberBlocks = Math.ceil(this.inputDimension.toDouble() / this.numberThreads.toDouble()).toInt()

    private val deviceForwardResult = Pointer()
    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private val deviceBackwardResult = Pointer()
    private val pointerToDeviceBackwardResult = Pointer.to(this.deviceBackwardResult)

    val deviceInputDimension = Pointer.to(intArrayOf(this.inputDimension))

    override fun acquire() {

        this.forwardKernel.acquire()
        this.backwardKernel.acquire()

        allocateDeviceMemory(this.deviceForwardResult, this.resultDimension)
        allocateDeviceMemory(this.deviceBackwardResult, this.inputDimension)

    }

    override fun forward(input : Pointer): Pointer {

        val forwardParameters = Pointer.to(
            this.deviceInputDimension,
            Pointer.to(input),
            this.pointerToDeviceForwardResult
        )

        this.forwardKernel.launch(forwardParameters, this.numberBlocks, this.numberThreads, 0)

        return this.deviceForwardResult

    }

    override fun backward(chain : Pointer) : Pointer {

        val backwardParameters = Pointer.to(
            this.deviceInputDimension,
            this.pointerToDeviceForwardResult,
            Pointer.to(chain),
            pointerToDeviceBackwardResult
        )

        this.backwardKernel.launch(backwardParameters, this.numberBlocks, this.numberThreads, 0)

        return this.deviceBackwardResult

    }

    override fun release() {

        this.forwardKernel.release()
        this.backwardKernel.release()

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceBackwardResult)

    }

}