package com.komputation.cpu.layers.continuation.convolution

import com.komputation.cpu.layers.BaseCpuHigherOrderContinuation
import com.komputation.cpu.layers.continuation.maxpooling.CpuMaxPooling
import com.komputation.cpu.layers.continuation.projection.CpuProjection
import com.komputation.optimization.Optimizable

class CpuConvolution internal constructor(
    name : String? = null,
    private val expansion: CpuExpansion,
    private val projection: CpuProjection,
    private val maxPooling: CpuMaxPooling) : BaseCpuHigherOrderContinuation(name, expansion, maxPooling), Optimizable {

    override val possibleInputLengths
        get() = this.expansion.possibleInputLengths
    override val possibleOutputLengths: IntArray
        get() = this.maxPooling.possibleOutputLengths

    fun getWeights() =
        this.projection.getWeights()

    fun getBias() =
        this.projection.getBias()

    override fun forward(withinBatch : Int, numberInputColumns : Int, input : FloatArray, isTraining : Boolean): FloatArray {
        this.expansion.forward(withinBatch, numberInputColumns, input, isTraining)

        this.projection.forward(withinBatch, this.expansion.numberOutputColumns, this.expansion.forwardResult, isTraining)

        this.maxPooling.forward(withinBatch, this.projection.numberOutputColumns, this.projection.forwardResult, isTraining)

        return this.forwardResult
    }

    override fun backward(withinBatch : Int, chain : FloatArray): FloatArray {
        this.maxPooling.backward(withinBatch, chain)

        this.projection.backward(withinBatch, this.maxPooling.backwardResult)

        this.expansion.backward(withinBatch, this.projection.backwardResult)

        return this.backwardResult
    }

    override fun optimize(batchSize : Int) {
        this.projection.optimize(batchSize)
    }

}