package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.functions.activation.differentiateSigmoid
import shape.komputation.cpu.functions.activation.sigmoid
import shape.komputation.cpu.functions.hadamard
import shape.komputation.cpu.layers.forward.dropout.DropoutCompliant
import shape.komputation.matrix.FloatMatrix

class CpuSigmoidLayer internal constructor(name : String? = null, private val numberEntries : Int) : BaseCpuActivationLayer(name), DropoutCompliant {

    private val forwardEntries = FloatArray(this.numberEntries)
    private val differentiation = FloatArray(this.numberEntries)
    private var hasCachedDifferentiation = false

    override fun forward(input : FloatMatrix, isTraining : Boolean): FloatMatrix {

        this.hasCachedDifferentiation = false

        sigmoid(input.entries, this.forwardEntries, this.numberEntries)

        val result = FloatMatrix(input.numberRows, input.numberColumns, this.forwardEntries)

        return result

    }

    override fun forward(input: FloatMatrix, mask: BooleanArray): FloatMatrix {

        this.hasCachedDifferentiation = false

        val inputEntries = input.entries

        for (index in 0..numberEntries - 1) {

            this.forwardEntries[index] = if(mask[index]) sigmoid(inputEntries[index]) else 0.0f

        }

        return FloatMatrix(input.numberRows, input.numberColumns, this.forwardEntries)

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

        return FloatMatrix(chain.numberRows, chain.numberColumns, hadamard(chain.entries, this.differentiation))

    }

}