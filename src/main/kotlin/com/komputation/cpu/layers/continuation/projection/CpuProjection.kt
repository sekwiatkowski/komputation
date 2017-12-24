package com.komputation.cpu.layers.continuation.projection

import com.komputation.cpu.layers.BaseCpuHigherOrderContinuation
import com.komputation.optimization.Optimizable

class CpuProjection internal constructor(
    name : String? = null,
    private val weighting: CpuWeighting,
    private val bias: CpuBias) : BaseCpuHigherOrderContinuation(name, weighting, bias), Optimizable {

    fun getWeights() =
        this.weighting.getWeights()

    fun getBias() =
        this.bias.getBias()

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining: Boolean): FloatArray {
        val weighted = this.weighting.forward(withinBatch, numberInputColumns, input, isTraining)

        this.bias.forward(withinBatch, numberInputColumns, weighted, isTraining)

        return this.forwardResult
    }

    override fun backward(withinBatch : Int, chain : FloatArray) : FloatArray {
        this.bias.backward(withinBatch, chain)
        this.weighting.backward(withinBatch, chain)

        return this.backwardResult
    }

    override fun optimize(batchSize : Int) {
        this.weighting.optimize(batchSize)
        this.bias.optimize(batchSize)
    }

}