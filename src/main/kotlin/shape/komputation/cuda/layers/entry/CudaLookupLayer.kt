package shape.komputation.cuda.layers.entry

import jcuda.Pointer
import shape.komputation.cpu.functions.concatenate
import shape.komputation.cpu.functions.pad
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

class CudaLookupLayer internal constructor(
    name : String?,
    private val vectors: Array<FloatArray>,
    maximumLength : Int,
    private val hasFixedLength: Boolean,
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

    private var numbersOfColumns = IntArray(0)
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

        this.numbersOfColumns = IntArray(maximumBatchSize)
        this.indices = IntArray(maximumBatchSize * this.maximumOutputColumns)

        this.maximumBatchSize[0] = maximumBatchSize

        this.kernel = this.createKernel()

        val concatenationSize = this.vectors.size * this.numberOutputRows
        val concatenation = FloatArray(concatenationSize)

        for ((index, vector) in this.vectors.withIndex()) {

            concatenate(vector, index * this.numberOutputRows, this.numberOutputRows, concatenation)

        }

        setFloatArray(concatenation, concatenationSize, this.deviceVectors)

        allocateDeviceFloatMemory(this.deviceForwardResult, maximumBatchSize * this.maximumOutputColumns * this.numberOutputRows)

        this.batchInputs = Array(maximumBatchSize) { IntArray(0) }

    }

    override fun release() {

        this.numbersOfColumns = IntArray(0)
        this.indices = IntArray(0)

        this.forwardResult = FloatArray(0)
        this.maximumBatchSize[0] = -1

        this.kernel!!.destroy()

    }

    override fun forward(batchId : Int, batchSize : Int, batch: IntArray, inputs : Array<Matrix>, memory: InputMemory) : Pointer {

        val optionalDeviceIndices = memory.tryToGetData(batchId)

        if (optionalDeviceIndices == null) {

            var batchNumberEntries = 0

            for ((withinBatch, id) in batch.withIndex()) {

                val input = inputs[id] as IntMatrix
                val inputEntries = input.entries

                val numberEntries = input.numberEntries

                this.batchInputs[withinBatch] = inputEntries
                this.numbersOfColumns[withinBatch] = numberEntries
                batchNumberEntries += numberEntries

                val finalEntries = if (this.hasFixedLength) {

                    inputEntries

                }
                else {

                    val paddedInputEntries = IntArray(this.maximumOutputColumns)
                    pad(inputEntries, numberEntries, this.maximumOutputColumns, -1, paddedInputEntries)

                    paddedInputEntries

                }

                concatenate(finalEntries, withinBatch * this.maximumOutputColumns, this.maximumOutputColumns, this.indices)

            }

            val deviceIndices = Pointer()
            setIntArray(this.indices, this.indices.size, deviceIndices)

            val deviceNumbersOfColumns = Pointer()
            setIntArray(this.numbersOfColumns, this.numbersOfColumns.size, deviceNumbersOfColumns)

            memory.setData(batchId, deviceIndices)
            memory.setTotalNumberOfColumns(batchId, batchNumberEntries)

            this.pointerToDeviceIndices = Pointer.to(deviceIndices)
            this.numberParameters = batchNumberEntries

        }
        else {

            this.pointerToDeviceIndices = Pointer.to(optionalDeviceIndices)
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

        this.updateRule?.sparseUpdate(this.numberParameters, this.pointerToDeviceIndices, this.pointerToDeviceVectors, scalingFactor, this.pointerToBackwardResult)

    }


}