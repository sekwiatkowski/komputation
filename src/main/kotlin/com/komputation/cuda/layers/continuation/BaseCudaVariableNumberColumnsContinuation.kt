package com.komputation.cuda.layers.continuation

import jcuda.Pointer

abstract class BaseCudaVariableNumberColumnsContinuation(
    name: String?,
    numberInputRows : Int,
    numberOutputRows : Int,
    maximumInputColumns : Int,
    private val computeNumberOutputColumns : (Int) -> Int) : BasePrimitiveCudaContinuation(name, numberInputRows, numberOutputRows, maximumInputColumns, computeNumberOutputColumns(maximumInputColumns)) {

    override fun computeOutputLengths(deviceInputLengths: Pointer, batchMaximumInputLength: Int) {
        this.largestNumberOutputColumnsInCurrentBatch = this.computeNumberOutputColumns(batchMaximumInputLength)
    }

}