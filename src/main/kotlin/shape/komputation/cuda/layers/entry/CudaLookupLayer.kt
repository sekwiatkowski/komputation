package shape.komputation.cuda.layers.entry

import jcuda.Pointer
import shape.komputation.cpu.functions.denselyConcatenateFloatArrays
import shape.komputation.cpu.functions.sparselyPadAndConcatenateIntMatrixEntries
import shape.komputation.cuda.CudaForwardState
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.cuda.kernels.Kernel
import shape.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import shape.komputation.cuda.layers.BaseCudaEntryPoint
import shape.komputation.cuda.memory.InputMemory
import shape.komputation.cuda.optimization.CudaUpdateRule
import shape.komputation.cuda.setFloatArray
import shape.komputation.cuda.setIntArray
import shape.komputation.layers.Resourceful
import shape.komputation.matrix.IntMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.optimization.Optimizable

class CudaLookupLayer(
    name : String?,
    private val vectors: Array<FloatArray>,
    maximumLength : Int,
    dimension : Int,
    private val updateRule: CudaUpdateRule?,
    private val createKernel: () -> Kernel,
    private val maximumNumberThreadsPerBlock : Int) : BaseCudaEntryPoint(name), CudaForwardState, Resourceful, Optimizable {

    override val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)
    private var pointerToBackwardResult = Pointer.to()

    override val numberOutputRows = dimension
    override val maximumOutputColumns = maximumLength

    private var pointerToDeviceIndices = Pointer()
    override var deviceNumberOutputColumns = Pointer()

    private var columnSizes = IntArray(0)
    private var indices = IntArray(0)
    private var forwardResult = FloatArray(0)
    private var maximumBatchSize = intArrayOf(-1)
    private var kernel : Kernel? = null

    private val deviceVectors = Pointer()
    private val pointerToDeviceVectors = Pointer.to(this.deviceVectors)

    private val launchConfiguration = computeColumnwiseLaunchConfiguration(
        this.numberOutputRows,
        this.maximumOutputColumns,
        this.maximumNumberThreadsPerBlock)

    private val pointerToMaximumBatchSize = Pointer.to(this.maximumBatchSize)
    private val pointerToMaximumLength = Pointer.to(intArrayOf(this.maximumOutputColumns))
    private val pointerToDimension = Pointer.to(intArrayOf(this.numberOutputRows))
    private val numberIterations = this.launchConfiguration.numberIterations
    private val pointerToNumberIterations = Pointer.to(intArrayOf(numberIterations))

    private var batchInputs = emptyArray<IntArray>()

    private var numberParameters = -1

    override fun acquire(maximumBatchSize: Int) {

        this.columnSizes = IntArray(maximumBatchSize)
        this.indices = IntArray(maximumBatchSize * this.maximumOutputColumns)

        this.maximumBatchSize[0] = maximumBatchSize

        this.kernel = this.createKernel()

        val concatenationSize = this.vectors.size * this.numberOutputRows
        val concatenation = FloatArray(concatenationSize)
        denselyConcatenateFloatArrays(this.vectors, this.numberOutputRows, concatenation)

        setFloatArray(concatenation, concatenationSize, this.deviceVectors)

        allocateDeviceFloatMemory(this.deviceForwardResult, maximumBatchSize * this.maximumOutputColumns * this.numberOutputRows)

        this.batchInputs = Array(maximumBatchSize) { IntArray(0) }

    }

    override fun release() {

        this.columnSizes = IntArray(0)
        this.indices = IntArray(0)

        this.forwardResult = FloatArray(0)
        this.maximumBatchSize[0] = -1

        this.kernel!!.destroy()

    }

    override fun forward(batchId : Int, batchSize : Int, batch: IntArray, inputs : Array<Matrix>, memory: InputMemory) : Pointer {

        val optionalDeviceIndices = memory.tryToGetData(batchId)

        if (optionalDeviceIndices == null) {

            var totalNumberOfColumns = 0

            for ((withinBatch, id) in batch.withIndex()) {

                val input = inputs[id] as IntMatrix

                val numberColumns = inputs[withinBatch].numberEntries.div(this.numberOutputRows)

                this.batchInputs[withinBatch] = input.entries
                this.columnSizes[withinBatch] = numberColumns
                totalNumberOfColumns += numberColumns

            }

            sparselyPadAndConcatenateIntMatrixEntries(this.batchInputs, this.maximumOutputColumns, this.indices)
            val deviceIndices = Pointer()
            setIntArray(this.indices, this.indices.size, deviceIndices)

            val deviceNumbersOfColumns = Pointer()
            setIntArray(this.columnSizes, this.columnSizes.size, deviceNumbersOfColumns)

            memory.setData(batchId, deviceIndices)
            memory.setColumnLengths(batchId, deviceNumbersOfColumns)
            memory.setTotalNumberOfColumns(batchId, totalNumberOfColumns)

            this.pointerToDeviceIndices = Pointer.to(deviceIndices)
            this.deviceNumberOutputColumns = this.deviceNumberOutputColumns
            this.numberParameters = totalNumberOfColumns

        }
        else {

            this.pointerToDeviceIndices = Pointer.to(optionalDeviceIndices)
            this.deviceNumberOutputColumns = memory.getDeviceNumbersOfColumns(batchId)
            this.numberParameters = memory.getTotalNumbersOfColumns(batchId)

        }

        this.kernel!!.launch(
            Pointer.to(
                this.pointerToDeviceVectors,
                this.pointerToDeviceIndices,
                this.pointerToForwardResult,
                this.pointerToMaximumBatchSize,
                this.pointerToMaximumLength,
                this.pointerToDimension,
                this.pointerToNumberIterations
            ),
            this.maximumBatchSize[0],
            this.launchConfiguration.numberBlocks,
            this.launchConfiguration.numberThreadsPerBlock,
            0)

        return this.deviceForwardResult

    }


    override fun backward(chain: Pointer) : Pointer {

        this.pointerToBackwardResult = Pointer.to(chain)

        return chain

    }

    override fun optimize(scalingFactor: Float) {

        // this.updateRule?.sparseUpdate(this.numberParameters, this.pointerToDeviceIndices, this.pointerToDeviceVectors, scalingFactor, this.pointerToBackwardResult)
        throw NotImplementedError()

    }


}