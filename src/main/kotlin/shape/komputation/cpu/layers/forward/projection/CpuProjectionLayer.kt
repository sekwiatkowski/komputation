package shape.komputation.cpu.layers.forward.projection

import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix
import shape.komputation.optimization.Optimizable

class CpuProjectionLayer internal constructor(
    name : String? = null,
    private val weightingLayer: CpuWeightingLayer,
    private val biasLayer : CpuBiasLayer?) : BaseCpuForwardLayer(name), Optimizable {

    override fun forward(withinBatch : Int, input: FloatMatrix, isTraining: Boolean): FloatMatrix {

        val weighted = this.weightingLayer.forward(withinBatch, input, isTraining)

        if (this.biasLayer != null) {

            val result = this.biasLayer.forward(withinBatch, weighted, isTraining)

            return result

        }
        else {

            return weighted

        }

    }

    override fun backward(withinBatch : Int, chain : FloatMatrix) : FloatMatrix {

        val backward = this.weightingLayer.backward(withinBatch, chain)

        this.biasLayer?.backward(withinBatch, chain)

        return backward

    }

    override fun optimize(scalingFactor : Float) {

        this.weightingLayer.optimize(scalingFactor)

        this.biasLayer?.optimize(scalingFactor)

    }

}