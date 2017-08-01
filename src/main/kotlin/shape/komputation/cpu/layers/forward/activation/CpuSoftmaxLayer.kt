package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.layers.forward.CpuNormalizationLayer
import shape.komputation.matrix.FloatMatrix

class CpuSoftmaxLayer internal constructor(
    name : String? = null,
    private val exponentiationLayer: CpuExponentiationLayer,
    private val normalizationLayer: CpuNormalizationLayer) : BaseCpuActivationLayer(name) {

    override fun forward(withinBatch : Int, input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        val exponentiation = this.exponentiationLayer.forward(withinBatch, input, isTraining)

        val normalization = this.normalizationLayer.forward(withinBatch, exponentiation, isTraining)

        return normalization

    }

    override fun backward(withinBatch : Int, chain : FloatMatrix): FloatMatrix {

        val backwardNormalization = this.normalizationLayer.backward(withinBatch, chain)

        val backwardExponentiation = this.exponentiationLayer.backward(withinBatch, backwardNormalization)

        return backwardExponentiation

    }

}