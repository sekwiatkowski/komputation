package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.functions.activation.differentiateSigmoid
import shape.komputation.cpu.functions.activation.sigmoid
import shape.komputation.cpu.functions.hadamard
import shape.komputation.matrix.FloatMatrix

class CpuSigmoidLayer internal constructor(name : String? = null, private val numberEntries : Int) : BaseCpuActivationLayer(name) {

    private val forwardEntries = FloatArray(this.numberEntries)
    private val differentiation = FloatArray(this.numberEntries)
    private val backwardEntries = FloatArray(this.numberEntries)
    private var hasCachedDifferentiation = false

    override fun forward(input : FloatMatrix, isTraining : Boolean): FloatMatrix {

        this.hasCachedDifferentiation = false

        sigmoid(input.entries, this.forwardEntries, this.numberEntries)

        val result = FloatMatrix(input.numberRows, input.numberColumns, this.forwardEntries)

        return result

    }

    /*
        input = pre-activation
        output = activation

        d activation / d pre-activation = activation * (1 - activation)
     */
    override fun backward(chain : FloatMatrix) : FloatMatrix {

        if (!this.hasCachedDifferentiation) {

            differentiateSigmoid(this.forwardEntries, this.differentiation, this.numberEntries)

            this.hasCachedDifferentiation = true

        }

        hadamard(chain.entries, this.differentiation, this.backwardEntries, this.numberEntries)

        return FloatMatrix(chain.numberRows, chain.numberColumns, this.backwardEntries)

    }

}