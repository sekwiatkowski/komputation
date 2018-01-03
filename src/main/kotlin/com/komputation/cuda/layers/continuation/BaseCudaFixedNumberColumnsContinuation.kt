package com.komputation.cuda.layers.continuation

import jcuda.Pointer

abstract class BaseCudaFixedNumberColumnsContinuation(
    name: String?,
    numberInputRows : Int,
    numberOutputRows : Int,
    maximumInputColumns : Int) : BasePrimitiveCudaContinuation(name, numberInputRows, numberOutputRows, maximumInputColumns, maximumInputColumns) {

    override var deviceForwardLengths = Pointer()

    override fun computeOutputLengths(deviceInputLengths : Pointer, batchMaximumInputLength: Int) {
        this.deviceForwardLengths = deviceInputLengths
        this.largestNumberOutputColumnsInCurrentBatch = batchMaximumInputLength
    }

}