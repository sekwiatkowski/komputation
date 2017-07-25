package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.layers.forward.CpuNormalizationLayer
import shape.komputation.matrix.FloatMatrix

class CpuSoftmaxLayer internal constructor(
    name : String? = null,
    private val exponentiationLayer: CpuExponentiationLayer,
    private val normalizationLayer: CpuNormalizationLayer) : BaseCpuActivationLayer(name) {

    override fun forward(input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        val exponentiation = this.exponentiationLayer.forward(input, isTraining)

        val normalization = this.normalizationLayer.forward(exponentiation, isTraining)

        return normalization

    }

    override fun backward(chain : FloatMatrix): FloatMatrix {

        val backwardNormalization = this.normalizationLayer.backward(chain)

        val backwardExponentiation = this.exponentiationLayer.backward(backwardNormalization)

        return backwardExponentiation

    }

}