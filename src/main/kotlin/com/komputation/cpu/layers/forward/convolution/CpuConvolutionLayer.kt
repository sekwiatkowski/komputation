package com.komputation.cpu.layers.forward.convolution

import com.komputation.cpu.layers.BaseCpuHigherOrderLayer
import com.komputation.cpu.layers.forward.maxpooling.CpuMaxPoolingLayer
import com.komputation.cpu.layers.forward.projection.CpuProjectionLayer
import com.komputation.optimization.Optimizable

class CpuConvolutionLayer internal constructor(
    name : String? = null,
    private val expansionLayer: CpuExpansionLayer,
    private val projectionLayer: CpuProjectionLayer,
    private val maxPoolingLayer: CpuMaxPoolingLayer) : BaseCpuHigherOrderLayer(name, expansionLayer.numberInputRows, maxPoolingLayer.numberOutputRows, expansionLayer, maxPoolingLayer), Optimizable {

    override val possibleInputLengths
        get() = this.expansionLayer.possibleInputLengths
    override val possibleOutputLengths: IntArray
        get() = this.maxPoolingLayer.possibleOutputLengths

    fun getWeights() =
        this.projectionLayer.getWeights()

    fun getBias() =
        this.projectionLayer.getBias()

    override fun forward(withinBatch : Int, numberInputColumns : Int, input : FloatArray, isTraining : Boolean): FloatArray {
        this.expansionLayer.forward(withinBatch, numberInputColumns, input, isTraining)

        this.projectionLayer.forward(withinBatch, this.expansionLayer.numberOutputColumns, this.expansionLayer.forwardResult, isTraining)

        this.maxPoolingLayer.forward(withinBatch, this.projectionLayer.numberOutputColumns, this.projectionLayer.forwardResult, isTraining)

        return this.forwardResult
    }

    override fun backward(withinBatch : Int, chain : FloatArray): FloatArray {
        this.maxPoolingLayer.backward(withinBatch, chain)

        this.projectionLayer.backward(withinBatch, this.maxPoolingLayer.backwardResult)

        this.expansionLayer.backward(withinBatch, this.projectionLayer.backwardResult)

        return this.backwardResult
    }

    override fun optimize(batchSize : Int) {
        this.projectionLayer.optimize(batchSize)
    }

}