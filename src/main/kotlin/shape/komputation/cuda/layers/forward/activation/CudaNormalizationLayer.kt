package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.Kernel
import shape.komputation.cuda.allocateDeviceMemory
import shape.komputation.layers.Resourceful

class CudaNormalizationLayer internal constructor(
    name : String? = null,
    private val forwardKernel: Kernel,
    private val numberRows : Int,
    private val numberColumns : Int) : BaseCudaActivationLayer(name), Resourceful {

    private val numberEntries = this.numberRows * this.numberColumns
    private val sharedMemoryBytes = (this.numberRows + this.numberRows / 2) * Sizeof.DOUBLE

    private val numberThreads = this.numberRows
    private val numberBlocks = this.numberColumns

    private val deviceForwardResult = Pointer()
    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private val pointerToNumberCategories = Pointer.to(intArrayOf(this.numberRows))

    override fun acquire() {

        this.forwardKernel.acquire()

        allocateDeviceMemory(this.deviceForwardResult, this.numberEntries)

    }

    override fun forward(input : Pointer): Pointer {

        val forwardParameters = Pointer.to(
            this.pointerToNumberCategories,
            Pointer.to(input),
            this.pointerToDeviceForwardResult
        )

        this.forwardKernel.launch(forwardParameters, this.numberBlocks, this.numberThreads, this.sharedMemoryBytes)

        return this.deviceForwardResult

    }

    override fun backward(chain : Pointer) : Pointer {

        TODO()

    }

    override fun release() {

        this.forwardKernel.release()

        cudaFree(this.deviceForwardResult)

    }

}