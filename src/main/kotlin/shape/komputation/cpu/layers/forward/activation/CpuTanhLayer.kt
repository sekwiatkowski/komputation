package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.functions.activation.differentiateTanh
import shape.komputation.cpu.functions.activation.tanh
import shape.komputation.cpu.functions.hadamard
import shape.komputation.matrix.FloatMath
import shape.komputation.matrix.FloatMatrix

class CpuTanhLayer internal constructor(name: String? = null, private val numberEntries : Int) : BaseCpuActivationLayer(name) {

    private val forwardEntries = FloatArray(this.numberEntries)
    private val backwardEntries = FloatArray(this.numberEntries)

    private var differentiation = FloatArray(this.numberEntries)
    private var hasCachedDifferentiation = false

    override fun forward(input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        this.hasCachedDifferentiation = false

        tanh(input.entries, this.forwardEntries, this.numberEntries)

        val result = FloatMatrix(input.numberRows, input.numberColumns, this.forwardEntries)

        return result
    }

    override fun backward(chain : FloatMatrix): FloatMatrix {

        if (!this.hasCachedDifferentiation) {

            differentiateTanh(this.forwardEntries, this.differentiation, this.numberEntries)

            this.hasCachedDifferentiation = true

        }

        hadamard(chain.entries, this.differentiation, this.backwardEntries, this.numberEntries)

        return FloatMatrix(chain.numberRows, chain.numberColumns, this.backwardEntries)

    }

}