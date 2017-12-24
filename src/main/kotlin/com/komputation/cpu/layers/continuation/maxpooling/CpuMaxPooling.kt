package com.komputation.cpu.layers.continuation.maxpooling

import com.komputation.cpu.functions.findMaxIndicesInRows
import com.komputation.cpu.functions.selectEntries
import com.komputation.cpu.layers.BaseCpuContinuationLayer
import java.util.*

class CpuMaxPooling internal constructor(
    name : String? = null,
    numberInputRows : Int,
    minimumInputColumns : Int,
    maximumInputColumns : Int) : BaseCpuContinuationLayer(name, numberInputRows, numberInputRows, minimumInputColumns, maximumInputColumns, { 1 }) {

    private val maxRowIndices = IntArray(this.numberInputRows)

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, forwardResult: FloatArray, isTraining: Boolean) {
        findMaxIndicesInRows(input, this.numberInputRows, numberInputColumns, this.maxRowIndices)

        selectEntries(input, this.maxRowIndices, forwardResult, this.numberInputRows)
    }

    override fun computeBackwardResult(withinBatch: Int, numberInputColumns: Int, numberOutputColumns : Int, forwardResult: FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        Arrays.fill(backwardResult, 0f)

        for (indexRow in 0 until this.numberInputRows) {
            backwardResult[this.maxRowIndices[indexRow]] = chain[indexRow]
        }
    }

}