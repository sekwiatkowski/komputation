package shape.komputation.cuda.layers.forward.expansion

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cpu.functions.computeNumberFilterColumnPositions
import shape.komputation.cpu.functions.computeNumberFilterRowPositions
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.cuda.computeDeviceFloatArraySize
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
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val warpSize : Int,
    maximumNumberThreads : Int) : BaseCudaForwardLayer(name), Resourceful, CudaVariableLengthForwardLayer {

    private val pointerToFilterHeight = Pointer.to(intArrayOf(filterHeight))
    private val pointerToFilterWidth = Pointer.to(intArrayOf(filterWidth))

    private val filterSize = filterHeight * filterWidth
    private val pointerToFilterSize = Pointer.to(intArrayOf(filterSize))
    override val numberOutputRows = this.filterSize

    override val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)

    override val deviceBackwardResult = Pointer()
    private val pointerToBackwardResult = Pointer.to(this.deviceForwardResult)

    private var forwardKernel : Kernel? = null
    private var backwardKernel : Kernel? = null

    private val batchSize = intArrayOf(-1)
    private val pointerToBatchSize = Pointer.to(this.batchSize)

    private val deviceMaximumBatchLengths = Pointer()
    private var pointerToMaximumBatchLengths = Pointer()

    private val pointerToNumberRows = Pointer.to(intArrayOf(this.numberInputRows))

    private val numberFilterRowPositions = computeNumberFilterRowPositions(this.numberInputRows, filterHeight)
    private val pointerToNumberFilterRowPositions = Pointer.to(intArrayOf(this.numberFilterRowPositions))
    private val numberFilterColumnPositions = computeNumberFilterColumnPositions(this.maximumInputColumns, filterHeight)

    private val maximumConvolutions = this.numberFilterRowPositions * this.numberFilterColumnPositions
    private val pointerToNumberConvolutions = Pointer.to(intArrayOf(this.maximumConvolutions))
    override val maximumOutputColumns = maximumConvolutions

    private val numberInputEntries = this.numberInputRows * this.maximumInputColumns
    private val pointerToNumberInputEntries = Pointer.to(intArrayOf(this.numberInputEntries))

    private val numberResultEntries = this.maximumOutputColumns * this.numberOutputRows
    private val pointerToNumberResultEntries = Pointer.to(intArrayOf(this.numberResultEntries))

    private var maximumBatchSize = -1

    private var numberForwardBlocksInXDimension = -1
    private val numberForwardBlocksInYDimension = this.maximumOutputColumns
    private val numberForwardThreads = this.numberOutputRows

    private var numberBackwardBlocksInXDimension = -1
    private val numberWarpsPerBlock = maximumNumberThreads / warpSize
    private val pointerToNumberWarpsPerBlock = Pointer.to(intArrayOf(this.numberWarpsPerBlock))
    private val numberBackwardBlocksInYDimension =
        if(this.numberInputEntries <= this.numberWarpsPerBlock)
            1
        else
            (this.numberInputEntries + numberWarpsPerBlock -1) / numberWarpsPerBlock
    private val numberBackwardThreads =
        if(this.numberBackwardBlocksInYDimension == 1)
            this.maximumInputColumns * warpSize
        else
            maximumNumberThreads


    override fun acquire(maximumBatchSize: Int) {

        this.maximumBatchSize = maximumBatchSize
        this.numberForwardBlocksInXDimension = this.maximumBatchSize
        this.numberBackwardBlocksInXDimension = this.maximumBatchSize

        this.forwardKernel = this.createForwardKernel()

        val maximumBatchLengths = IntArray(maximumBatchSize) { this.maximumInputColumns }
        setIntArray(maximumBatchLengths, maximumBatchSize, this.deviceMaximumBatchLengths)

        this.pointerToMaximumBatchLengths = Pointer.to(this.deviceMaximumBatchLengths)

        allocateDeviceFloatMemory(this.deviceForwardResult, maximumBatchSize * this.numberResultEntries)

        this.backwardKernel = this.createBackwardKernel()

        allocateDeviceFloatMemory(this.deviceBackwardResult, maximumBatchSize * this.numberInputEntries)

    }

    override fun release() {

        cudaFree(this.deviceMaximumBatchLengths)

        this.forwardKernel!!.destroy()
        cudaFree(this.deviceForwardResult)

        this.backwardKernel!!.destroy()
        cudaFree(this.deviceBackwardResult)

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
            this.numberForwardBlocksInXDimension,
            this.numberForwardBlocksInYDimension,
            this.numberForwardThreads,
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
            this.numberForwardBlocksInXDimension,
            this.numberForwardBlocksInYDimension,
            this.numberForwardThreads,
            0
        )

        return this.deviceForwardResult

    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {

        this.backwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToMaximumBatchLengths,
                this.pointerToNumberRows,
                this.pointerToNumberInputEntries,
                this.pointerToNumberWarpsPerBlock,
                this.pointerToFilterHeight,
                this.pointerToFilterWidth,
                this.pointerToFilterSize,
                this.pointerToNumberConvolutions,
                this.pointerToNumberFilterRowPositions,
                Pointer.to(chain),
                this.pointerToBackwardResult
            ),
            this.numberBackwardBlocksInXDimension,
            this.numberBackwardBlocksInYDimension,
            this.numberBackwardThreads,
            computeDeviceFloatArraySize(this.warpSize).toInt()
        )

        return this.deviceBackwardResult

    }


}