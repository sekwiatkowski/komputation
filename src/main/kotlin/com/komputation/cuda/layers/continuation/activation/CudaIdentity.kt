package com.komputation.cuda.layers.continuation.activation

import com.komputation.cuda.layers.continuation.BaseCudaContinuation
import com.komputation.cuda.layers.continuation.CudaActivation
import jcuda.Pointer

class CudaIdentity internal constructor(name : String? = null, numberRows : Int, numberColumns : Int) : BaseCudaContinuation(name, numberRows, numberRows, numberColumns, numberColumns), CudaActivation  {

    override var deviceForwardResult = Pointer()
    override var deviceBackwardResult = Pointer()
    override var deviceForwardLengths = Pointer()

    override fun forward(batchSize: Int, deviceInput: Pointer, deviceInputLengths : Pointer, isTraining: Boolean): Pointer {
        this.deviceForwardResult = deviceInput
        this.deviceForwardLengths = deviceInputLengths

        return this.deviceForwardResult
    }

    override fun backward(batchSize: Int, chain: Pointer) : Pointer {
        this.deviceBackwardResult = chain

        return this.deviceBackwardResult
    }

}