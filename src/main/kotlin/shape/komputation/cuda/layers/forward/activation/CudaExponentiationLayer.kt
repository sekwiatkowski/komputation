package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.Kernel
import shape.komputation.cuda.allocateDeviceMemory
import shape.komputation.layers.Resourceful

class CudaExponentiationLayer internal constructor(
    name : String? = null,
    private val forwardExponentiationKernel: Kernel,
    private val backwardExponentiationKernel : Kernel,
    numberRows : Int,
    numberColumns : Int) : BaseCudaActivationLayer(name), Resourceful {

    private val numberThreadsPerBlock = numberRows
    private val numberBlocks = numberColumns

    private val numberEntries = numberRows * numberColumns

    private val deviceForwardResult = Pointer()
    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private val deviceBackwardResult = Pointer()
    private val pointerToDeviceBackwardResult = Pointer.to(this.deviceBackwardResult)

    override fun acquire() {

        allocateDeviceMemory(this.deviceForwardResult, this.numberEntries)

        this.forwardExponentiationKernel.acquire()

        allocateDeviceMemory(this.deviceBackwardResult, this.numberEntries)

        this.backwardExponentiationKernel.acquire()

    }

    override fun forward(input : Pointer): Pointer {

        this.forwardExponentiationKernel.launch(
            Pointer.to(
                Pointer.to(input),
                this.pointerToDeviceForwardResult
            ),
            this.numberBlocks,
            this.numberThreadsPerBlock,
            0
        )

        return this.deviceForwardResult

    }

    override fun backward(chain : Pointer) : Pointer {

        this.backwardExponentiationKernel.launch(
            Pointer.to(
                this.pointerToDeviceForwardResult,
                Pointer.to(chain),
                this.pointerToDeviceBackwardResult
            ),
            this.numberBlocks,
            this.numberThreadsPerBlock,
            0
        )

        return this.deviceBackwardResult

    }

    override fun release() {

        this.backwardExponentiationKernel.release()

        cudaFree(this.deviceBackwardResult)

        this.forwardExponentiationKernel.release()

        cudaFree(this.deviceForwardResult)

    }

}