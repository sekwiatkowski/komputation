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
    private val backwardKernel: Kernel,
    private val numberRows : Int,
    private val numberColumns : Int) : BaseCudaActivationLayer(name), Resourceful {

    private val numberEntries = this.numberRows * this.numberColumns

    private val numberThreads = this.numberRows
    private val numberBlocks = this.numberColumns

    private val forwardSharedMemoryBytes = (this.numberRows + this.numberRows / 2) * Sizeof.DOUBLE
    private val backwardSharedMemoryBytes = Sizeof.DOUBLE

    private val pointerToNumberCategories = Pointer.to(intArrayOf(this.numberRows))

    private val deviceForwardResult = Pointer()
    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private val deviceSums = Pointer()
    private val pointerToDeviceSums = Pointer.to(this.deviceSums)

    private val deviceBackwardResult = Pointer()
    private val pointerToDeviceBackwardResult = Pointer.to(this.deviceBackwardResult)

    override fun acquire() {

        allocateDeviceMemory(this.deviceForwardResult, this.numberEntries)
        allocateDeviceMemory(this.deviceSums, this.numberBlocks)

        this.forwardKernel.acquire()

        allocateDeviceMemory(this.deviceBackwardResult, this.numberEntries)

        this.backwardKernel.acquire()

    }

    private var pointerToInput = Pointer()

    override fun forward(input : Pointer): Pointer {

        val pointerToInput = Pointer.to(input)

        val forwardParameters = Pointer.to(
            this.pointerToNumberCategories,
            pointerToInput,
            this.pointerToDeviceForwardResult,
            this.pointerToDeviceSums
        )

        this.pointerToInput = pointerToInput

        this.forwardKernel.launch(forwardParameters, this.numberBlocks, this.numberThreads, this.forwardSharedMemoryBytes)

        return this.deviceForwardResult

    }

    private val backwardParameters = Pointer.to(
        this.pointerToNumberCategories,
        this.pointerToInput,
        this.pointerToDeviceBackwardResult
    )

    override fun backward(chain : Pointer) : Pointer {

        this.backwardKernel.launch(this.backwardParameters, this.numberBlocks, this.numberThreads, this.backwardSharedMemoryBytes)

        return this.deviceBackwardResult

    }

    override fun release() {

        cudaFree(this.deviceBackwardResult)

        this.backwardKernel.release()

        this.forwardKernel.release()

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceSums)

    }

}