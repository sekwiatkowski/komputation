package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.runtime.JCuda
import shape.komputation.cuda.Kernel
import shape.komputation.cuda.allocateDeviceMemory

abstract class BaseCudaEntrywiseActivationLayer internal constructor(
    name : String? = null,
    private val forwardKernel: Kernel,
    private val backwardKernel: Kernel,
    maximumThreadsPerBlock: Int,
    private val numberEntries : Int) : BaseCudaActivationLayer(name) {

    private val numberThreads = Math.min(this.numberEntries, maximumThreadsPerBlock)
    private val numberBlocks = Math.ceil(this.numberEntries.toDouble() / this.numberThreads.toDouble()).toInt()

    private val deviceForwardResult = Pointer()
    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private val deviceBackwardResult = Pointer()
    private val pointerToDeviceBackwardResult = Pointer.to(this.deviceBackwardResult)

    private val deviceInputDimension = Pointer.to(intArrayOf(this.numberEntries))

    override fun acquire() {

        allocateDeviceMemory(this.deviceForwardResult, this.numberEntries)

        this.forwardKernel.acquire()

        allocateDeviceMemory(this.deviceBackwardResult, this.numberEntries)

        this.backwardKernel.acquire()

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

        JCuda.cudaFree(this.deviceForwardResult)
        JCuda.cudaFree(this.deviceBackwardResult)

    }

}