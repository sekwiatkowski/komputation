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
    numberRows: Int,
    numberColumns : Int) : BaseCudaEntryPoint(name, numberRows, numberColumns), Resourceful {

    override var deviceForwardResult = Pointer()
    override var deviceForwardLengths = Pointer()

    private var data = FloatArray(0)
    private var lengths = IntArray(0)

    override fun acquire(maximumBatchSize: Int) {
        super.acquire(maximumBatchSize)

        this.data = FloatArray(this.maximumBatchOutputEntries)
        this.lengths = IntArray(this.maximumBatchOutputEntries)
    }

    override fun release() {
        super.release()

        this.data = FloatArray(0)
        this.lengths = IntArray(0)
    }

    override fun forward(
        batchId : Int,
        batchSize : Int,
        batch: IntArray,
        inputs : Array<out Matrix>,
        memory: InputMemory): Pointer {

        val data = memory.tryToGetData(batchId)

        if (data == null) {
            Arrays.fill(this.data, Float.NaN)

            for ((withinBatch, id) in batch.withIndex()) {
                val input = inputs[id] as FloatMatrix

                val inputEntries = input.entries
                val length = input.numberColumns

                copy(inputEntries, withinBatch * this.maximumOutputEntries, inputEntries.size, this.data)
                this.lengths[withinBatch] = length
            }

            val deviceForwardResult = Pointer()
            setFloatArray(this.data, this.maximumBatchOutputEntries, deviceForwardResult)
            this.deviceForwardResult = deviceForwardResult

            val deviceForwardLengths = Pointer()
            setIntArray(this.lengths, batchSize, deviceForwardLengths)
            this.deviceForwardLengths = deviceForwardLengths

            memory.setVariableLengthData(batchId, deviceForwardResult, deviceForwardLengths)
        }
        else {
            this.deviceForwardResult = data
            this.deviceForwardLengths = memory.getDeviceLengths(batchId)
        }

        return this.deviceForwardResult
    }

    override fun backward(batchId: Int, chain: Pointer, memory: InputMemory) =
        chain

}