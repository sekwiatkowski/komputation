package com.komputation.cuda.layers.continuation.activation

import com.komputation.cuda.layers.continuation.CudaActivation
import jcuda.Pointer

class CudaIdentity internal constructor(val name : String? = null, numberRows : Int, numberColumns : Int) : CudaActivation {

    override val batchMaximumOutputColumns: Int
        get() = this.batchMaximumInputColumns
    override var deviceForwardResult = Pointer()
    override val numberInputRows = numberRows
    override val maximumInputColumns = numberColumns

    override var batchMaximumInputColumns = -1
    override var deviceBackwardResult = Pointer()
    override val numberOutputRows = numberRows
    override val maximumOutputColumns = numberColumns

    override var deviceForwardLengths = Pointer()

    override fun forward(batchSize: Int, deviceInput: Pointer, deviceInputLengths : Pointer, batchMaximumInputLength: Int, isTraining: Boolean): Pointer {
        this.deviceForwardResult = deviceInput
        this.deviceForwardLengths = deviceInputLengths
        this.batchMaximumInputColumns = batchMaximumInputLength

        return this.deviceForwardResult
    }

    override fun backward(batchSize: Int, chain: Pointer) : Pointer {

        this.deviceBackwardResult = chain

        return this.deviceBackwardResult
    }

}