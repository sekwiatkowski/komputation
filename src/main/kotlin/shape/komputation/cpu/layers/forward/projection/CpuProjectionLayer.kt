package shape.komputation.cpu.layers.forward.projection

import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.layers.Resourceful
import shape.komputation.optimization.Optimizable

class CpuProjectionLayer internal constructor(
    name : String? = null,
    private val weightingLayer: CpuWeightingLayer,
    private val biasLayer : CpuBiasLayer?) : BaseCpuForwardLayer(name), Resourceful, Optimizable {

    override val numberOutputRows = this.weightingLayer.numberOutputRows
    override var numberOutputColumns = -1
    override var forwardResult = FloatArray(0)

    override val numberInputRows = this.weightingLayer.numberInputRows
    override var numberInputColumns = -1
    override var backwardResult = FloatArray(0)

    override fun acquire(maximumBatchSize: Int) {

        this.weightingLayer.acquire(maximumBatchSize)
        this.biasLayer?.acquire(maximumBatchSize)

    }

    override fun release() {

        this.weightingLayer.release()
        this.biasLayer?.release()

    }

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining: Boolean): FloatArray {

        this.numberInputColumns = numberInputColumns

        val weighted = this.weightingLayer.forward(withinBatch, numberInputColumns, input, isTraining)

        this.numberOutputColumns = this.weightingLayer.numberOutputColumns

        this.forwardResult = if (this.biasLayer != null) {

            this.biasLayer.forward(withinBatch, numberInputColumns, weighted, isTraining)

        }
        else {

            weighted

        }

        return this.forwardResult

    }

    override fun backward(withinBatch : Int, chain : FloatArray) : FloatArray {

        this.weightingLayer.backward(withinBatch, chain)

        this.biasLayer?.backward(withinBatch, chain)

        this.backwardResult = this.weightingLayer.backwardResult

        return this.backwardResult

    }

    override fun optimize(scalingFactor : Float) {

        this.weightingLayer.optimize(scalingFactor)

        this.biasLayer?.optimize(scalingFactor)

    }

}