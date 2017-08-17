package shape.komputation.cuda.layers.forward.dense

import jcuda.Pointer
import shape.komputation.cuda.layers.BaseCudaForwardLayer
import shape.komputation.cuda.layers.forward.activation.CudaActivationLayer
import shape.komputation.cuda.layers.forward.projection.CublasProjectionLayer
import shape.komputation.optimization.Optimizable

class CudaDenseLayer internal constructor(
    name: String?,
    private val projectionLayer: CublasProjectionLayer,
    private val activationLayer: CudaActivationLayer) : BaseCudaForwardLayer(name), Optimizable {

    override val deviceForwardResult
        get() = this.activationLayer.deviceForwardResult
    override val numberOutputRows
        get() = this.activationLayer.numberOutputRows
    override val numberOutputColumns
        get() = this.activationLayer.numberOutputColumns

    override val deviceBackwardResult
        get() = this.projectionLayer.deviceForwardResult
    override val numberInputRows
        get() = this.projectionLayer.numberInputRows
    override val numberInputColumns
        get() = this.projectionLayer.numberInputColumns


    override fun forward(batchSize: Int, numberInputColumns : Int, input: Pointer, isTraining: Boolean): Pointer {

        val projected = this.projectionLayer.forward(batchSize, numberInputColumns, input, isTraining)

        val activated = this.activationLayer.forward(batchSize, this.projectionLayer.numberOutputColumns, projected, isTraining)

        return activated

    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {

        val backwardActivation = this.activationLayer.backward(batchSize, chain)

        val backwardProjection = this.projectionLayer.backward(batchSize, backwardActivation)

        return backwardProjection

    }

    override fun optimize(scalingFactor: Float) {

        this.projectionLayer.optimize(scalingFactor)

    }

}