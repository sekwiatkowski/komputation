package com.komputation.cpu.layers.continuation.dense

import com.komputation.cpu.layers.BaseCpuHigherOrderContinuation
import com.komputation.cpu.layers.continuation.activation.CpuActivation
import com.komputation.cpu.layers.continuation.projection.CpuProjection
import com.komputation.optimization.Optimizable

class CpuDense internal constructor(
    name : String?,
    private val projection : CpuProjection,
    private val activation: CpuActivation) : BaseCpuHigherOrderContinuation(name, projection, activation), Optimizable {

    fun getBias() =
        this.projection.getBias()

    fun getWeights() =
        this.projection.getWeights()

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean): FloatArray {
        val projected = this.projection.forward(withinBatch, numberInputColumns, input, isTraining)

        val activated = this.activation.forward(withinBatch, numberInputColumns, projected, isTraining)

        return activated
    }

    override fun backward(withinBatch : Int, chain: FloatArray): FloatArray {
        this.activation.backward(withinBatch, chain)

        val backward = this.projection.backward(withinBatch, this.activation.backwardResult)

        return backward
    }

    override fun optimize(batchSize : Int) {
        this.projection.optimize(batchSize)
    }

}