package shape.komputation.cuda.layers.forward

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.Kernel
import shape.komputation.cuda.allocateDeviceMemory
import shape.komputation.cuda.computeDeviceByteSize
import shape.komputation.cuda.layers.forward.activation.BaseCudaActivationLayer
import shape.komputation.layers.Resourceful

class CudaNormalizationLayer internal constructor(
    name : String? = null,
    private val forwardKernel: Kernel,
    private val backwardKernel: Kernel,
    private val blockSize : Int,
    private val numberRows : Int,
    private val numberColumns : Int) : BaseCudaActivationLayer(name), Resourceful {

    private val numberEntries = this.numberRows * this.numberColumns

    private val reductionLength = this.numberRows + this.numberRows / 2
    private val forwardSharedMemoryBytes = computeDeviceByteSize(this.reductionLength).toInt()
    private val backwardSharedMemoryBytes = computeDeviceByteSize(1 + this.reductionLength).toInt()

    private val pointerToNumberCategories = Pointer.to(intArrayOf(this.numberRows))

    private val deviceForwardResult = Pointer()
    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private val deviceSums = Pointer()
    private val pointerToDeviceSums = Pointer.to(this.deviceSums)

    private val deviceBackwardResult = Pointer()
    private val pointerToDeviceBackwardResult = Pointer.to(this.deviceBackwardResult)

    override fun acquire() {

        allocateDeviceMemory(this.deviceForwardResult, this.numberEntries)
        allocateDeviceMemory(this.deviceSums, this.numberColumns)

        this.forwardKernel.acquire()

        allocateDeviceMemory(this.deviceBackwardResult, this.numberEntries)

        this.backwardKernel.acquire()

    }

    private var pointerToDeviceInput = Pointer()

    override fun forward(input : Pointer, isTraining : Boolean): Pointer {

        this.pointerToDeviceInput = Pointer.to(input)

        val forwardParameters = Pointer.to(
            this.pointerToNumberCategories,
            this.pointerToDeviceInput,
            this.pointerToDeviceForwardResult,
            this.pointerToDeviceSums
        )

        this.forwardKernel.launch(forwardParameters, this.numberColumns, this.blockSize, this.forwardSharedMemoryBytes)

        return this.deviceForwardResult

    }


    override fun backward(chain : Pointer) : Pointer {

        this.backwardKernel.launch(
            Pointer.to(
                this.pointerToNumberCategories,
                Pointer.to(chain),
                this.pointerToDeviceForwardResult,
                this.pointerToDeviceSums,
                this.pointerToDeviceBackwardResult
            ),
            this.numberColumns,
            this.blockSize,
            this.backwardSharedMemoryBytes)

        return this.deviceBackwardResult

    }

    override fun release() {

        cudaFree(this.deviceBackwardResult)

        this.backwardKernel.release()

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceSums)

        this.forwardKernel.release()

    }

}