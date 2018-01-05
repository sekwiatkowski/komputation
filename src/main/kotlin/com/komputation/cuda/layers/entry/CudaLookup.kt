package com.komputation.cuda.layers.entry

import com.komputation.cpu.functions.copy
import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import com.komputation.cuda.kernels.launch.computeNumberSegments
import com.komputation.cuda.layers.BaseCudaEntryPoint
import com.komputation.cuda.memory.DuplicateMemory
import com.komputation.cuda.memory.InputMemory
import com.komputation.cuda.optimization.BaseCudaUpdateRule
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setIntArray
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
    private val createGroupSumKernel: () -> Kernel,
    private val maximumNumberThreadsPerBlock : Int) : BaseCudaEntryPoint(name, maximumInputLength, dimension), Optimizable {

    private val numberVectorsEntries = vectors.size
    private val deviceVectors = Pointer()
    private val pointerToVectors = Pointer.to(this.deviceVectors)

    private var maximumParametersInBatch = -1

    override val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)

    private var lengths = IntArray(0)

    private var indices = IntArray(0)
    private var deviceIndices = Pointer()
    private var pointerToIndices = Pointer()

    private var forwardResult = FloatArray(0)
    override var deviceForwardLengths = Pointer()
    override var largestNumberOutputColumnsInCurrentBatch = if(this.hasFixedLength) this.maximumOutputColumns else -1

    private var forwardKernel: Kernel? = null
    private var groupSumKernel: Kernel? = null

    private val launchConfiguration = computeColumnwiseLaunchConfiguration(
        this.numberOutputRows,
        this.maximumOutputColumns,
        this.maximumNumberThreadsPerBlock)

    private val pointerToMaximumLength = Pointer.to(intArrayOf(this.maximumOutputColumns))
    private val pointerToDimension = Pointer.to(intArrayOf(this.numberOutputRows))

    private val forwardNumberIterations = this.launchConfiguration.numberIterations
    private val pointerToForwardNumberIterations = Pointer.to(intArrayOf(this.forwardNumberIterations))

    private val deviceMaximumForwardLengths = Pointer()

    private val groupSumNumberIterations = computeNumberSegments(dimension, this.maximumNumberThreadsPerBlock)
    private val pointerToGroupSumNumberIterations = Pointer.to(intArrayOf(this.groupSumNumberIterations))
    private val groupSumNumberThreadsThreads = if(this.groupSumNumberIterations == 1) dimension else this.maximumNumberThreadsPerBlock

    override fun acquire(maximumBatchSize: Int) {

        super.acquire(maximumBatchSize)

        this.maximumParametersInBatch = maximumBatchSize * this.maximumOutputColumns

        allocateDeviceFloatMemory(this.deviceForwardResult, this.maximumBatchOutputEntries)

        this.forwardKernel = this.createForwardKernel()
        this.groupSumKernel = this.createGroupSumKernel()

        this.lengths = IntArray(maximumBatchSize)
        this.indices = IntArray(this.maximumParametersInBatch)

        setFloatArray(this.vectors, this.numberVectorsEntries, this.deviceVectors)

        if (this.hasFixedLength) {
            val maximumForwardLengths = IntArray(maximumBatchSize) { this.maximumOutputColumns }
            setIntArray(maximumForwardLengths, maximumBatchSize, this.deviceMaximumForwardLengths)
            this.deviceForwardLengths = this.deviceMaximumForwardLengths
        }
    }

    override fun release() {
        this.maximumParametersInBatch = -1

        this.forwardResult = FloatArray(0)
        cudaFree(this.deviceForwardResult)

        this.forwardKernel!!.destroy()
        this.groupSumKernel!!.destroy()

        this.lengths = IntArray(0)
        this.indices = IntArray(0)

        this.vectors = getFloatArray(this.deviceVectors, this.numberVectorsEntries)
        cudaFree(this.deviceVectors)
    }

    override fun forward(batchId : Int, batchSize : Int, batch: IntArray, inputs : Array<Matrix>, memory: InputMemory) : Pointer {
        val maximumBatchSize = this.maximumBatchSize

        val data = memory.tryToGetData(batchId)

        if (data != null) {
            this.deviceIndices = memory.getData(batchId)

            if (this.hasFixedLength) {
                this.largestNumberOutputColumnsInCurrentBatch = this.maximumOutputColumns
            }
            else {
                this.deviceForwardLengths = memory.getDeviceLengths(batchId)
                this.largestNumberOutputColumnsInCurrentBatch = memory.getMaximumLength(batchId)
            }
        }
        else {
            Arrays.fill(this.indices, 0, this.maximumParametersInBatch, -1)
            Arrays.fill(this.lengths, 0, this.maximumBatchSize, -0)
            var maximumLength = -1

            val occurrences = hashMapOf<Int, ArrayList<Pair<Int, Int>>>()

            for ((withinBatch, id) in batch.withIndex()) {
                val input = (inputs[id] as IntMatrix)
                val inputEntries = input.entries
                val length = input.numberEntries

                val firstIndexWithinBatch = withinBatch * this.maximumOutputColumns

                copy(inputEntries, firstIndexWithinBatch, length, this.indices)
                this.lengths[withinBatch] = length
                maximumLength = Math.max(length, maximumLength)

                for ((withinInstance, inputEntry) in inputEntries.withIndex()) {
                    if (!occurrences.containsKey(inputEntry)) {
                        occurrences[inputEntry] = arrayListOf()
                    }
                    occurrences[inputEntry]!!.add(withinBatch to withinInstance)
                }

            }

            val deviceIndices = Pointer()
            setIntArray(this.indices, this.maximumParametersInBatch, deviceIndices)
            this.deviceIndices = deviceIndices

            if (this.hasFixedLength) {
                memory.setFixedLengthData(batchId, this.deviceIndices)
            }
            else {
                val deviceForwardLengths = Pointer()
                setIntArray(this.lengths, this.maximumBatchSize, deviceForwardLengths)
                this.deviceForwardLengths = deviceForwardLengths

                this.largestNumberOutputColumnsInCurrentBatch = maximumLength

                memory.setVariableLengthData(batchId, deviceIndices, deviceForwardLengths, maximumLength)
            }

            val duplicates = occurrences.filter { (_, indices) -> indices.size > 1 }
            val numberDuplicates = duplicates.size

            val counts = IntArray(this.maximumParametersInBatch)
            occurrences.values.forEach { list ->
                val (withinBatch, withinInstance) = list.first()
                counts[withinBatch * this.maximumOutputColumns + withinInstance] = 1
            }

            if (numberDuplicates > 0) {
                val numberOtherOccurrences = duplicates.map { (_, indices) -> indices.size - 1 }.sum()

                val firstOccurrences = IntArray(numberDuplicates)
                val otherOccurrences = IntArray(numberOtherOccurrences)
                val otherOccurrencePositions = IntArray(numberDuplicates + 1)

                var previousEnd = 0

                duplicates.values.forEachIndexed { indexDuplicate, indices ->

                    val (firstOccurrenceInstanceIndex, firstOccurrenceIndexWithinInstance) = indices.first()
                    val firstOccurrenceIndexWithinBatch = firstOccurrenceInstanceIndex * this.maximumOutputColumns + firstOccurrenceIndexWithinInstance

                    val remainingIndices = indices.drop(1)
                    val currentNumberOtherOccurrences = remainingIndices.size

                    val startPosition = previousEnd
                    val exclusiveEndPosition = previousEnd + currentNumberOtherOccurrences

                    val visitedInstances = hashSetOf(firstOccurrenceInstanceIndex)

                    remainingIndices.forEachIndexed { index, toBeAdded ->

                        val (otherOccurrenceInstanceIndex, otherOccurrenceIndexWithinInstance) = toBeAdded
                        val otherOccurrenceIndexWithinBatch = otherOccurrenceInstanceIndex * this.maximumOutputColumns + otherOccurrenceIndexWithinInstance

                        otherOccurrences[startPosition + index] = otherOccurrenceIndexWithinBatch

                        if (!visitedInstances.contains(otherOccurrenceInstanceIndex)) {
                            counts[firstOccurrenceIndexWithinBatch] = counts[firstOccurrenceIndexWithinBatch] + 1
                            counts[otherOccurrenceIndexWithinBatch] = 0
                            visitedInstances.add(otherOccurrenceInstanceIndex)
                        }

                    }

                    firstOccurrences[indexDuplicate] = firstOccurrenceIndexWithinBatch
                    otherOccurrencePositions[indexDuplicate + 1] = exclusiveEndPosition

                    previousEnd = exclusiveEndPosition
                }

                val deviceFirstOccurrences = Pointer()
                setIntArray(firstOccurrences, numberDuplicates, deviceFirstOccurrences)
                val deviceOtherOccurrences = Pointer()
                setIntArray(otherOccurrences, numberOtherOccurrences, deviceOtherOccurrences)
                val deviceOtherOccurrencePositions = Pointer()
                setIntArray(otherOccurrencePositions, otherOccurrencePositions.size, deviceOtherOccurrencePositions)

                memory.setWithDuplicates(batchId, numberDuplicates, DuplicateMemory(deviceFirstOccurrences, deviceOtherOccurrences, deviceOtherOccurrencePositions))

            }
            else {
                memory.setWithoutDuplicates(batchId)
            }

            val deviceCounts = Pointer()
            setIntArray(counts, this.maximumParametersInBatch, deviceCounts)
            memory.setCounts(batchId, deviceCounts)

        }

        this.pointerToIndices = Pointer.to(this.deviceIndices)

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToVectors,
                this.pointerToIndices,
                this.pointerToForwardResult,
                this.pointerToMaximumBatchSize,
                this.pointerToMaximumLength,
                this.pointerToDimension,
                this.pointerToForwardNumberIterations
            ),
            maximumBatchSize,
            this.launchConfiguration.numberBlocks,
            this.launchConfiguration.numberThreadsPerBlock,
            0)

        return this.deviceForwardResult
    }

    private var pointerToCounts = Pointer()
    private var pointerToGradient = Pointer()

    override fun backward(batchId : Int, chain: Pointer, memory: InputMemory) : Pointer {

        this.pointerToGradient = Pointer.to(chain)

        val numberDuplicates = memory.getNumberDuplicates(batchId)
        val deviceCounts = memory.getDeviceCounts(batchId)
        this.pointerToCounts = Pointer.to(deviceCounts)

        if (numberDuplicates > 0) {
            val (deviceFirstOccurrences, deviceOtherOccurrences, deviceOccurrencePositions) = memory.getDuplicateMemory(batchId)

            this.groupSumKernel!!.launch(
                Pointer.to(
                    this.pointerToGradient,
                    Pointer.to(deviceFirstOccurrences),
                    Pointer.to(deviceOtherOccurrences),
                    Pointer.to(deviceOccurrencePositions),
                    this.pointerToDimension,
                    this.pointerToGroupSumNumberIterations
                ),
                numberDuplicates,
                1,
                this.groupSumNumberThreadsThreads,
                0)

        }

        return chain
    }

    override fun optimize(batchSize: Int) {

        this.updateRule?.sparseUpdate(
            this.maximumParametersInBatch,
            this.pointerToIndices,
            this.pointerToCounts,
            this.pointerToVectors,
            this.pointerToGradient)

    }

}