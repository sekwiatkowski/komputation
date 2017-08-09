package shape.komputation.cpu.layers.forward.convolution

import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.forward.projection.CpuProjectionLayer
import shape.komputation.layers.Resourceful
import shape.komputation.optimization.Optimizable

class CpuConvolutionalLayer internal constructor(
    name : String? = null,
    override val numberInputRows : Int,
    private val expansionLayer: CpuExpansionLayer,
    private val projectionLayer: CpuProjectionLayer,
    private val maxPoolingLayer: CpuMaxPoolingLayer) : BaseCpuForwardLayer(name), Resourceful, Optimizable {

    override fun acquire(maximumBatchSize: Int) {

        this.expansionLayer.acquire(maximumBatchSize)
        this.projectionLayer.acquire(maximumBatchSize)
        this.maxPoolingLayer.acquire(maximumBatchSize)

    }

    override fun release() {

        this.expansionLayer.release()
        this.projectionLayer.release()
        this.maxPoolingLayer.release()

    }

    override val numberOutputRows = this.numberInputRows
    override val numberOutputColumns = 1
    override var forwardResult = FloatArray(0)

    override var numberInputColumns = -1
    override var backwardResult = FloatArray(0)

    override fun forward(withinBatch : Int, numberInputColumns : Int, input : FloatArray, isTraining : Boolean): FloatArray {

        this.numberInputColumns = numberInputColumns

        this.expansionLayer.forward(withinBatch, numberInputColumns, input, isTraining)

        this.projectionLayer.forward(withinBatch, this.expansionLayer.numberOutputColumns, this.expansionLayer.forwardResult, isTraining)

        this.forwardResult = this.maxPoolingLayer.forward(withinBatch, this.projectionLayer.numberOutputColumns, this.projectionLayer.forwardResult, isTraining)

        return this.forwardResult

    }

    override fun backward(withinBatch : Int, chain : FloatArray): FloatArray {

        this.maxPoolingLayer.backward(withinBatch, chain)

        this.projectionLayer.backward(withinBatch, this.maxPoolingLayer.backwardResult)

        this.backwardResult = this.expansionLayer.backward(withinBatch, this.projectionLayer.backwardResult)

        return this.backwardResult

    }

    override fun optimize(scalingFactor : Float) {

        this.projectionLayer.optimize(scalingFactor)

    }

}