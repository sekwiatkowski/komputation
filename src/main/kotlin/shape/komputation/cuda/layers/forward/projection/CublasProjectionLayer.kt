package shape.komputation.cuda.layers.forward.projection

import jcuda.Pointer
import shape.komputation.cuda.layers.BaseCudaForwardLayer
import shape.komputation.optimization.Optimizable

class CublasProjectionLayer internal constructor(
    name: String?,
    private val weightingLayer: CublasWeightingLayer,
    private val biasLayer: CublasBiasLayer? = null) : BaseCudaForwardLayer(name), Optimizable {

    override var deviceForwardResult = Pointer()
    override var numberOutputRows = -1
    override var numberOutputColumns = -1

    override var deviceBackwardResult = Pointer()
    override var numberInputRows = -1
    override var numberInputColumns = -1

    override fun forward(batchSize: Int, numberInputColumns : Int, input: Pointer, isTraining: Boolean): Pointer {

        val weighted = this.weightingLayer.forward(batchSize, numberInputColumns, input, isTraining)

        if (this.biasLayer == null) {

            this.deviceForwardResult = this.weightingLayer.deviceForwardResult
            this.numberOutputRows = this.weightingLayer.numberOutputRows
            this.numberOutputColumns = this.weightingLayer.numberOutputColumns

            return weighted

        }
        else {

            val weightedAndBiased = this.biasLayer.forward(batchSize, this.numberOutputColumns, weighted, isTraining)

            this.deviceBackwardResult = this.weightingLayer.deviceBackwardResult
            this.numberInputRows = this.weightingLayer.numberInputRows
            this.numberInputColumns = this.weightingLayer.numberInputColumns

            return weightedAndBiased

        }

    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {

        val backwardWeighting = this.weightingLayer.backward(batchSize, chain)

        if (this.biasLayer != null) {

            this.biasLayer.backward(batchSize, chain)

        }

        return backwardWeighting

    }

    override fun optimize(scalingFactor: Float) {

        this.weightingLayer.optimize(scalingFactor)

        this.biasLayer?.optimize(scalingFactor)

    }

}