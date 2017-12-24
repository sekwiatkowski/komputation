package com.komputation.cuda.layers.continuation.dense

import jcuda.Pointer
import com.komputation.cuda.layers.continuation.BaseCudaHigherOrderContinuation
import com.komputation.cuda.layers.continuation.CudaActivation
import com.komputation.cuda.layers.continuation.projection.CublasProjection
import com.komputation.optimization.Optimizable

class CudaDense internal constructor(
    name: String?,
    private val projection: CublasProjection,
    private val activation: CudaActivation) : BaseCudaHigherOrderContinuation(name, projection, activation), Optimizable {

    fun getDeviceWeights() =
        this.projection.getDeviceWeights()

    fun getDeviceBias() =
        this.projection.getDeviceBias()

    override fun forward(batchSize: Int, deviceInput: Pointer, deviceInputLengths : Pointer, batchMaximumInputLength: Int, isTraining: Boolean): Pointer {
        val projected = this.projection.forward(batchSize, deviceInput, deviceInputLengths, batchMaximumInputLength, isTraining)

        val activated = this.activation.forward(batchSize, projected, this.projection.deviceForwardLengths, this.projection.batchMaximumOutputColumns, isTraining)

        return activated
    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {

        val backwardActivation = this.activation.backward(batchSize, chain)

        val backwardProjection = this.projection.backward(batchSize, backwardActivation)

        return backwardProjection
    }

    override fun optimize(batchSize: Int) {
        this.projection.optimize(batchSize)
    }

}