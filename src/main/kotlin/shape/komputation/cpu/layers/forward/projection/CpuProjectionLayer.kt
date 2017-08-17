package shape.komputation.cpu.layers.forward.projection

import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.optimization.Optimizable

class CpuProjectionLayer internal constructor(
    name : String? = null,
    private val weightingLayer: CpuWeightingLayer,
    private val biasLayer : CpuBiasLayer?) : BaseCpuForwardLayer(name), Optimizable {

    override val numberOutputRows
        get() = this.weightingLayer.numberOutputRows
    override val numberOutputColumns
        get() = this.weightingLayer.numberOutputColumns
    override var forwardResult = FloatArray(0)

    override val numberInputRows
        get() = this.weightingLayer.numberInputRows
    override val numberInputColumns
        get() = this.weightingLayer.numberInputColumns
    override var backwardResult = FloatArray(0)

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining: Boolean): FloatArray {

        val weighted = this.weightingLayer.forward(withinBatch, numberInputColumns, input, isTraining)

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