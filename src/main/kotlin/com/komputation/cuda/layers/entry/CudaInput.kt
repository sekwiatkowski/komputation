package com.komputation.cuda.layers.entry

import com.komputation.cpu.functions.copy
import com.komputation.cuda.layers.BaseCudaEntryPoint
import com.komputation.cuda.memory.InputMemory
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setIntArray
import com.komputation.instructions.Resourceful
import com.komputation.matrix.FloatMatrix
import com.komputation.matrix.Matrix
import jcuda.Pointer
import java.util.*

class CudaInput internal constructor(
    name : String?,
    private val numberRows: Int,
    private val numberColumns : Int) : BaseCudaEntryPoint(name), Resourceful {

    override var deviceForwardResult = Pointer()
    override var deviceForwardLengths = Pointer()
    override var largestNumberOutputColumnsInCurrentBatch = -1

    private val maximumInstanceEntries = this.numberRows * this.numberColumns
    private var maximumBatchSize = -1
    private var maximumBatchEntries = -1
    private var data = FloatArray(0)
    private var lengths = IntArray(0)

    override fun acquire(maximumBatchSize: Int) {
        this.maximumBatchSize = maximumBatchSize
        this.maximumBatchEntries = maximumBatchSize * this.maximumInstanceEntries
        this.data = FloatArray(this.maximumBatchEntries)
        this.lengths = IntArray(this.maximumBatchEntries)
    }

    override fun release() {
        this.data = FloatArray(0)
        this.maximumBatchEntries = -1
        this.maximumBatchSize = -1
    }

    override fun forward(
        batchId : Int,
        batchSize : Int,
        batch: IntArray,
        inputs : Array<Matrix>,
        memory: InputMemory): Pointer {

        val data = memory.tryToGetData(batchId)

        if (data == null) {
            Arrays.fill(this.data, Float.NaN)

            var maximumLength = -1

            for ((withinBatch, id) in batch.withIndex()) {
                val input = inputs[id] as FloatMatrix

                val inputEntries = input.entries
                val length = input.numberColumns

                copy(inputEntries, withinBatch * this.maximumInstanceEntries, inputEntries.size, this.data)
                this.lengths[withinBatch] = length
                maximumLength = Math.max(length, maximumLength)
            }

            val deviceForwardResult = Pointer()
            setFloatArray(this.data, this.maximumBatchEntries, deviceForwardResult)
            this.deviceForwardResult = deviceForwardResult

            val deviceForwardLengths = Pointer()
            setIntArray(this.lengths, batchSize, deviceForwardLengths)
            this.deviceForwardLengths = deviceForwardLengths

            this.largestNumberOutputColumnsInCurrentBatch = maximumLength

            memory.set(batchId, deviceForwardResult, deviceForwardLengths, maximumLength)
        }
        else {
            this.deviceForwardResult = data
            this.deviceForwardLengths = memory.getDeviceLengths(batchId)
            this.largestNumberOutputColumnsInCurrentBatch = memory.getHostMaximumLength(batchId)
        }

        return this.deviceForwardResult
    }

    override fun backward(chain: Pointer) =
        chain

}