package shape.komputation.cpu.layers.forward.projection

import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix
import shape.komputation.optimization.Optimizable

class CpuProjectionLayer internal constructor(
    name : String? = null,
    private val weightingLayer: CpuWeightingLayer,
    private val biasLayer : CpuBiasLayer?) : BaseCpuForwardLayer(name), Optimizable {

    override fun forward(input: FloatMatrix, isTraining: Boolean): FloatMatrix {

        val weighted = this.weightingLayer.forward(input, isTraining)

        if (this.biasLayer != null) {

            return this.biasLayer.forward(weighted, isTraining)

        }
        else {

            return weighted

        }

    }

    override fun backward(chain : FloatMatrix) : FloatMatrix {

        val backward = this.weightingLayer.backward(chain)

        this.biasLayer?.backward(chain)

        return backward

    }

    override fun optimize(scalingFactor : Float) {

        this.weightingLayer.optimize(scalingFactor)

        this.biasLayer?.optimize(scalingFactor)

    }

}