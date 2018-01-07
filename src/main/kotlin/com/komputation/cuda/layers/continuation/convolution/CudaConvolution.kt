package com.komputation.cuda.layers.continuation.convolution

import com.komputation.cuda.layers.continuation.BaseCudaHigherOrderContinuation
import com.komputation.cuda.layers.continuation.maxpooling.CudaMaxPooling
import com.komputation.cuda.layers.continuation.projection.CublasProjection
import com.komputation.optimization.Optimizable
import jcuda.Pointer

class CudaConvolution(
    name : String?,
    private val expansion: CudaExpansion,
    private val projection: CublasProjection,
    private val maxPooling: CudaMaxPooling) : BaseCudaHigherOrderContinuation(name, expansion, maxPooling), Optimizable {

    fun getDeviceWeights() =
        this.projection.getDeviceWeights()

    fun getDeviceBias() =
        this.projection.getDeviceBias()

    override fun forward(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, isTraining: Boolean): Pointer {
        val expanded = this.expansion.forward(batchSize, deviceInput, deviceInputLengths, isTraining)

        val projected = this.projection.forward(batchSize, expanded, this.expansion.deviceForwardLengths, isTraining)

        val maxPooled = this.maxPooling.forward(batchSize, projected, this.projection.deviceForwardLengths, isTraining)

        return maxPooled
    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {
        val backwardMaxPooling = this.maxPooling.backward(batchSize, chain)

        val backwardProjection = this.projection.backward(batchSize, backwardMaxPooling)

        val backwardExpansion = this.expansion.backward(batchSize, backwardProjection)

        return backwardExpansion
    }

    override fun optimize(batchSize: Int) {
        this.projection.optimize(batchSize)
    }

}