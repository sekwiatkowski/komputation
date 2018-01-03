package com.komputation.cuda.layers.entry

import com.komputation.cpu.functions.copy
import com.komputation.cuda.*
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import com.komputation.cuda.layers.BaseCudaEntryPoint
import com.komputation.cuda.memory.InputMemory
import com.komputation.cuda.optimization.BaseCudaUpdateRule
import com.komputation.instructions.Resourceful
import com.komputation.matrix.IntMatrix
import com.komputation.matrix.Matrix
import com.komputation.optimization.Optimizable
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import java.util.*

class CudaLookup internal constructor(
    name : String?,
    private var vectors: FloatArray,
    maximumInputLength: Int,
    private val hasFixedLength: Boolean,
    dimension : Int,
    private val updateRule: BaseCudaUpdateRule?,
    private val createForwardKernel: () -> Kernel,
    private val hashing: CudaHashing,
    private val groupSum: CudaGroupSum,
    private val maximumNumberThreadsPerBlock : Int) : BaseCudaEntryPoint(name), CudaForwardState, Resourceful, Optimizable {

    private val numberVectorsEntries = vectors.size
    private val deviceVectors = Pointer()
    private val pointerToDeviceVectors = Pointer.to(this.deviceVectors)

    override val numberOutputRows = dimension
    override val maximumOutputColumns = maximumInputLength
    override val maximumOutputEntries = this.numberOutputRows * this.maximumOutputColumns
    private var maximumNumberParametersInBatch = -1
    private var maximumBatchEntries = -1

    override val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)

    private var lengths = IntArray(0)

    private var indices = IntArray(0)
    private var deviceIndices = Pointer()
    private var pointerToIndices = Pointer.to(this.deviceIndices)

    private var forwardResult = FloatArray(0)
    override var deviceForwardLengths = Pointer()
    override var largestNumberOutputColumnsInCurrentBatch = if(this.hasFixedLength) this.maximumOutputColumns else -1

    private var maximumBatchSize = intArrayOf(-1)

    private var forwardKernel: Kernel? = null

    private val launchConfiguration = computeColumnwiseLaunchConfiguration(
        this.numberOutputRows,
        this.maximumOutputColumns,
        this.maximumNumberThreadsPerBlock)

    private val pointerToMaximumBatchSize = Pointer.to(this.maximumBatchSize)
    private val pointerToMaximumLength = Pointer.to(intArrayOf(this.maximumOutputColumns))
    private val pointerToDimension = Pointer.to(intArrayOf(this.numberOutputRows))
    private val numberIterations = this.launchConfiguration.numberIterations
    private val pointerToNumberIterations = Pointer.to(intArrayOf(this.numberIterations))

    private val deviceMaximumForwardLengths = Pointer()

    override fun acquire(maximumBatchSize: Int) {
        this.maximumBatchSize[0] = maximumBatchSize
        this.maximumNumberParametersInBatch = maximumBatchSize * this.maximumOutputColumns

        this.maximumBatchEntries = maximumBatchSize * this.maximumOutputEntries
        allocateDeviceFloatMemory(this.deviceForwardResult, this.maximumBatchEntries)

        this.forwardKernel = this.createForwardKernel()

        this.lengths = IntArray(maximumBatchSize)
        this.indices = IntArray(this.maximumNumberParametersInBatch)

        setFloatArray(this.vectors, this.numberVectorsEntries, this.deviceVectors)

        this.hashing.acquire(maximumBatchSize)
        this.groupSum.acquire(maximumBatchSize)

        if (this.hasFixedLength) {
            val maximumForwardLengths = IntArray(maximumBatchSize) { this.maximumOutputColumns }
            setIntArray(maximumForwardLengths, maximumBatchSize, this.deviceMaximumForwardLengths)
            this.deviceForwardLengths = this.deviceMaximumForwardLengths
        }
    }

    override fun release() {
        this.maximumBatchSize[0] = -1

        this.maximumNumberParametersInBatch = -1

        this.forwardResult = FloatArray(0)
        cudaFree(this.deviceForwardResult)

        this.forwardKernel!!.destroy()

        this.lengths = IntArray(0)
        this.indices = IntArray(0)

        this.vectors = getFloatArray(this.deviceVectors, this.numberVectorsEntries)
        cudaFree(this.deviceVectors)

        this.groupSum.release()
        this.hashing.release()
    }

    override fun forward(batchId : Int, batchSize : Int, batch: IntArray, inputs : Array<Matrix>, memory: InputMemory) : Pointer {
        val maximumBatchSize = this.maximumBatchSize[0]

        val data = memory.tryToGetData(batchId)

        if (data != null) {
            this.deviceIndices = memory.getData(batchId)

            if (this.hasFixedLength) {
                this.largestNumberOutputColumnsInCurrentBatch = this.maximumOutputColumns
            }
            else {
                this.deviceForwardLengths = memory.getDeviceLengths(batchId)
                this.largestNumberOutputColumnsInCurrentBatch = memory.getHostMaximumLength(batchId)
            }
        }
        else {
            Arrays.fill(this.indices, 0, this.maximumNumberParametersInBatch, -1)
            Arrays.fill(this.lengths,0, this.maximumBatchSize[0], -0)
            var maximumLength = -1
            for ((withinBatch, id) in batch.withIndex()) {
                val input = inputs[id] as IntMatrix
                val inputEntries = input.entries
                val length = input.numberEntries

                copy(inputEntries, withinBatch * this.maximumOutputColumns, length, this.indices)
                this.lengths[withinBatch] = length
                maximumLength = Math.max(length, maximumLength)
            }

            val deviceIndices = Pointer()
            setIntArray(this.indices, this.maximumNumberParametersInBatch, deviceIndices)
            this.deviceIndices = deviceIndices

            if (this.hasFixedLength) {
                memory.setData(batchId, this.deviceIndices)
            }
            else {
                val deviceForwardLengths = Pointer()
                setIntArray(this.lengths, this.maximumBatchSize[0], deviceForwardLengths)
                this.deviceForwardLengths = deviceForwardLengths

                this.largestNumberOutputColumnsInCurrentBatch = maximumLength

                memory.set(batchId, deviceIndices, deviceForwardLengths, maximumLength)
            }
        }

        this.pointerToIndices = Pointer.to(this.deviceIndices)

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToDeviceVectors,
                this.pointerToIndices,
                this.pointerToForwardResult,
                this.pointerToMaximumBatchSize,
                this.pointerToMaximumLength,
                this.pointerToDimension,
                this.pointerToNumberIterations
            ),
            maximumBatchSize,
            this.launchConfiguration.numberBlocks,
            this.launchConfiguration.numberThreadsPerBlock,
            0)

        return this.deviceForwardResult
    }

    override fun backward(chain: Pointer) : Pointer {
        this.hashing.reset()
        this.hashing.hash(this.pointerToIndices)

        this.groupSum.reset()
        this.groupSum.sum(this.hashing.getPointerToMapping(), Pointer.to(chain))

        return chain
    }

    override fun optimize(batchSize: Int) {
        this.updateRule?.sparseUpdate(
            this.hashing.getHashTableSize(),
            this.hashing.getPointerToHashTable(),
            this.hashing.getPointerToCounts(),
            this.pointerToDeviceVectors,
            this.groupSum.getPointerToSum())
    }

}