package com.komputation.cuda.layers.continuation

import com.komputation.cuda.layers.BaseCudaContinuation
import jcuda.Pointer

abstract class BaseCudaFixedNumberColumnsContinuation(
    name: String?,
    numberInputRows : Int,
    numberOutputRows : Int,
    maximumInputColumns : Int) : BaseCudaContinuation(name, numberInputRows, numberOutputRows, maximumInputColumns, maximumInputColumns) {

    override var deviceForwardLengths = Pointer()

    override fun computeOutputLengths(deviceInputLengths : Pointer, batchMaximumInputLength: Int) {
        this.deviceForwardLengths = deviceInputLengths
        this.batchMaximumOutputColumns = batchMaximumInputLength
    }

}