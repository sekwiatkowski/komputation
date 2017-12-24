package com.komputation.cuda.layers.continuation.projection

import com.komputation.cuda.layers.continuation.BaseCudaHigherOrderContinuation
import com.komputation.optimization.Optimizable
import jcuda.Pointer

class CublasProjection internal constructor(
    name: String?,
    private val weighting: CublasWeighting,
    private val bias: CudaBias) : BaseCudaHigherOrderContinuation(name, weighting, bias), Optimizable {

    fun getDeviceWeights() =
        this.weighting.getDeviceWeights()

    fun getDeviceBias() =
        this.bias.getDeviceBias()

    override fun forward(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, batchMaximumInputLength: Int, isTraining: Boolean): Pointer {
        val weighted = this.weighting.forward(batchSize, deviceInput, deviceInputLengths, batchMaximumInputLength, isTraining)

        this.bias.forward(batchSize, weighted, this.weighting.deviceForwardLengths, this.weighting.batchMaximumOutputColumns, isTraining)

        return this.deviceForwardResult
    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {
        this.bias.backward(batchSize, chain)

        this.weighting.backward(batchSize, chain)

        return this.deviceBackwardResult
    }

    override fun optimize(batchSize: Int) {
        this.weighting.optimize(batchSize)

        this.bias.optimize(batchSize)
    }

}