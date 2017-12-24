package com.komputation.cuda.layers.continuation

import com.komputation.cuda.layers.BaseCudaContinuation
import jcuda.Pointer

abstract class BaseCudaChangingNumberColumnsContinuation(
    name: String?,
    numberInputRows : Int,
    numberOutputRows : Int,
    maximumInputColumns : Int,
    private val computeNumberOutputColumns : (Int) -> Int) : BaseCudaContinuation(name, numberInputRows, numberOutputRows, maximumInputColumns, computeNumberOutputColumns(maximumInputColumns)) {

    override fun computeOutputLengths(deviceInputLengths: Pointer, batchMaximumInputLength: Int) {
        this.batchMaximumOutputColumns = this.computeNumberOutputColumns(batchMaximumInputLength)
    }

}