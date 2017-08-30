package shape.komputation.cuda.layers.forward.convolution

import jcuda.Pointer
import shape.komputation.cuda.layers.BaseCudaForwardLayer
import shape.komputation.cuda.layers.forward.maxpooling.CudaMaxPoolingLayer
import shape.komputation.cuda.layers.forward.projection.CublasProjectionLayer
import shape.komputation.optimization.Optimizable

class CudaConvolutionLayer(
    name : String?,
    private val expansionLayer: CudaExpansionLayer,
    private val projectionLayer: CublasProjectionLayer,
    private val maxPoolingLayer: CudaMaxPoolingLayer) : BaseCudaForwardLayer(name), Optimizable {

    override val deviceForwardResult: Pointer
        get() = this.maxPoolingLayer.deviceForwardResult
    override val numberOutputRows: Int
        get() = this.maxPoolingLayer.numberOutputRows
    override val maximumOutputColumns: Int
        get() = this.maxPoolingLayer.maximumOutputColumns

    override val deviceBackwardResult: Pointer
        get() = this.expansionLayer.deviceBackwardResult
    override val numberInputRows: Int
        get() = this.expansionLayer.numberInputRows
    override val maximumInputColumns: Int
        get() = this.expansionLayer.maximumInputColumns

    override fun forward(batchSize: Int, deviceInput: Pointer, isTraining: Boolean) : Pointer {

        val expanded = this.expansionLayer.forward(batchSize, deviceInput, isTraining)

        val projected = this.projectionLayer.forward(batchSize, expanded, isTraining)

        val maxPooled = this.maxPoolingLayer.forward(batchSize, projected, isTraining)

        return maxPooled

    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {

        val backwardMaxPooling = this.maxPoolingLayer.backward(batchSize, chain)

        val backwardProjection = this.projectionLayer.backward(batchSize, backwardMaxPooling)

        val backwardExpansion = this.expansionLayer.backward(batchSize, backwardProjection)

        return backwardExpansion

    }

    override fun optimize(scalingFactor : Float) {

        this.projectionLayer.optimize(scalingFactor)

    }

}