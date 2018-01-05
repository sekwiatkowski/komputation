package com.komputation.cuda.layers.continuation.activation

import com.komputation.cuda.layers.continuation.BaseCudaHigherOrderContinuation
import com.komputation.cuda.layers.continuation.CudaActivation
import com.komputation.cuda.layers.continuation.normalization.CudaNormalization
import jcuda.Pointer

class CudaSoftmax internal constructor(
    name : String? = null,
    private val exponentiation: CudaExponentiation,
    private val normalization: CudaNormalization) : BaseCudaHigherOrderContinuation(name, exponentiation, normalization), CudaActivation {

    override fun forward(batchSize: Int, deviceInput: Pointer, deviceInputLengths : Pointer, largestNumberInputColumnsInBatch: Int, isTraining: Boolean): Pointer {
        val exponentiated = this.exponentiation.forward(batchSize, deviceInput, deviceInputLengths, largestNumberInputColumnsInBatch, isTraining)

        val normalized = this.normalization.forward(batchSize, exponentiated, this.exponentiation.deviceForwardLengths, this.exponentiation.largestNumberOutputColumnsInCurrentBatch, isTraining)

        return normalized
    }

    override fun backward(batchSize: Int, chain: Pointer) : Pointer {

        val backwardNormalization = this.normalization.backward(batchSize, chain)

        val backwardExponentiation = this.exponentiation.backward(batchSize, backwardNormalization)

        return backwardExponentiation
    }

}