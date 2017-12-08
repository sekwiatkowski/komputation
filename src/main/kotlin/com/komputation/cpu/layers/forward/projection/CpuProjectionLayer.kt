package com.komputation.cpu.layers.forward.projection

import com.komputation.cpu.layers.BaseCpuHigherOrderLayer
import com.komputation.layers.Resourceful
import com.komputation.optimization.Optimizable

class CpuProjectionLayer internal constructor(
    name : String? = null,
    private val weightingLayer: CpuWeightingLayer,
    private val biasLayer : CpuBiasLayer) : BaseCpuHigherOrderLayer(name, weightingLayer, biasLayer), Resourceful, Optimizable {

    fun getWeights() =
        this.weightingLayer.getWeights()

    fun getBias() =
        this.biasLayer.getBias()

    override fun acquire(maximumBatchSize: Int) {
        this.weightingLayer.acquire(maximumBatchSize)
        this.biasLayer.acquire(maximumBatchSize)
    }

    override fun release() {
        this.weightingLayer.release()
        this.biasLayer.release()
    }

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining: Boolean): FloatArray {
        val weighted = this.weightingLayer.forward(withinBatch, numberInputColumns, input, isTraining)

        this.biasLayer.forward(withinBatch, numberInputColumns, weighted, isTraining)

        return this.forwardResult
    }

    override fun backward(withinBatch : Int, chain : FloatArray) : FloatArray {
        this.biasLayer.backward(withinBatch, chain)
        this.weightingLayer.backward(withinBatch, chain)

        return this.backwardResult
    }

    override fun optimize(batchSize : Int) {
        this.weightingLayer.optimize(batchSize)
        this.biasLayer.optimize(batchSize)
    }

}