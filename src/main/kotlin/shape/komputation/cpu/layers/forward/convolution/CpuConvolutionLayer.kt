package shape.komputation.cpu.layers.forward.convolution

import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.forward.maxpooling.CpuMaxPoolingLayer
import shape.komputation.cpu.layers.forward.projection.CpuProjectionLayer
import shape.komputation.optimization.Optimizable

class CpuConvolutionLayer internal constructor(
    name : String? = null,
    private val expansionLayer: CpuExpansionLayer,
    private val projectionLayer: CpuProjectionLayer,
    private val maxPoolingLayer: CpuMaxPoolingLayer) : BaseCpuForwardLayer(name), Optimizable {

    override val numberOutputRows
        get() = this.maxPoolingLayer.numberOutputRows
    override val numberOutputColumns
        get() = this.maxPoolingLayer.numberOutputColumns
    override val forwardResult
        get() = this.maxPoolingLayer.forwardResult

    override val numberInputRows: Int
        get() = this.expansionLayer.numberInputRows
    override val numberInputColumns
        get() = this.expansionLayer.numberInputColumns
    override val backwardResult
        get() = this.expansionLayer.backwardResult

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

    override fun optimize(scalingFactor : Float) {

        this.projectionLayer.optimize(scalingFactor)

    }

}