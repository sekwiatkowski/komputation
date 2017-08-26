package shape.komputation.cuda.layers.forward.expansion

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cpu.functions.computeNumberFilterColumnPositions
import shape.komputation.cpu.functions.computeNumberFilterRowPositions
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.cuda.kernels.Kernel
import shape.komputation.cuda.layers.BaseCudaForwardLayer
import shape.komputation.cuda.layers.CudaVariableLengthForwardLayer
import shape.komputation.cuda.setIntArray
import shape.komputation.layers.Resourceful

class CudaExpansionLayer internal constructor(
    name : String?,
    override val numberInputRows : Int,
    override val maximumInputColumns : Int,
    filterHeight : Int,
    filterWidth : Int,
    private val createForwardKernel: () -> Kernel) : BaseCudaForwardLayer(name), Resourceful, CudaVariableLengthForwardLayer {

    private val pointerToFilterHeight = Pointer.to(intArrayOf(filterHeight))
    private val pointerToFilterWidth = Pointer.to(intArrayOf(filterWidth))

    private val filterSize = filterHeight * filterWidth
    private val pointerToFilterSize = Pointer.to(intArrayOf(filterSize))
    override val numberOutputRows = this.filterSize

    override val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)

    override val deviceBackwardResult = Pointer()

    private var forwardKernel : Kernel? = null

    private val batchSize = intArrayOf(-1)
    private val pointerToBatchSize = Pointer.to(this.batchSize)

    private val deviceMaximumBatchLengths = Pointer()
    private var pointerToMaximumBatchLengths = Pointer()

    private val pointerToNumberRows = Pointer.to(intArrayOf(this.numberInputRows))

    private val numberFilterRowPositions = computeNumberFilterRowPositions(this.numberInputRows, filterHeight)
    private val pointerToNumberFilterRowPositions = Pointer.to(intArrayOf(this.numberFilterRowPositions))
    private val numberFilterColumnPositions = computeNumberFilterColumnPositions(this.maximumInputColumns, filterHeight)
    override val maximumOutputColumns = this.numberFilterRowPositions * this.numberFilterColumnPositions

    private val numberInputEntries = this.numberInputRows * this.maximumInputColumns
    private val pointerToNumberInputEntries = Pointer.to(intArrayOf(this.numberInputEntries))

    private val numberResultEntries = this.maximumOutputColumns * this.numberOutputRows
    private val pointerToNumberResultEntries = Pointer.to(intArrayOf(this.numberResultEntries))

    private var maximumBatchSize = -1

    override fun acquire(maximumBatchSize: Int) {

        this.maximumBatchSize = maximumBatchSize

        this.forwardKernel = this.createForwardKernel()

        val maximumBatchLengths = IntArray(maximumBatchSize) { this.maximumInputColumns }
        setIntArray(maximumBatchLengths, maximumBatchSize, this.deviceMaximumBatchLengths)

        this.pointerToMaximumBatchLengths = Pointer.to(this.deviceMaximumBatchLengths)

        allocateDeviceFloatMemory(this.deviceForwardResult, maximumBatchSize * this.numberResultEntries)

    }

    override fun release() {

        this.forwardKernel!!.destroy()

        cudaFree(this.deviceMaximumBatchLengths)
        cudaFree(this.deviceForwardResult)

    }

    override fun forward(batchSize: Int, deviceInput: Pointer, isTraining: Boolean): Pointer {

        this.batchSize[0] = batchSize

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToMaximumBatchLengths,
                this.pointerToNumberRows,
                this.pointerToNumberFilterRowPositions,
                this.pointerToNumberInputEntries,
                this.pointerToNumberResultEntries,
                this.pointerToFilterHeight,
                this.pointerToFilterWidth,
                this.pointerToFilterSize,
                Pointer.to(deviceInput),
                this.pointerToForwardResult
            ),
            this.maximumBatchSize,
            this.maximumOutputColumns,
            this.numberOutputRows,
            0
        )

        return this.deviceForwardResult

    }

    override fun forward(batchSize: Int, deviceLengths: Pointer, deviceInput: Pointer, isTraining: Boolean) : Pointer {

        this.batchSize[0] = batchSize

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                Pointer.to(deviceLengths),
                this.pointerToNumberRows,
                this.pointerToNumberFilterRowPositions,
                this.pointerToNumberInputEntries,
                this.pointerToNumberResultEntries,
                this.pointerToFilterHeight,
                this.pointerToFilterWidth,
                this.pointerToFilterSize,
                Pointer.to(deviceInput),
                this.pointerToForwardResult
            ),
            this.maximumBatchSize,
            this.maximumOutputColumns,
            this.numberOutputRows,
            0
        )

        return this.deviceForwardResult

    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {

        return this.deviceBackwardResult

    }


}