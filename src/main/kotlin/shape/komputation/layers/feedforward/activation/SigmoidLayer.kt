package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.differentiateSigmoid
import shape.komputation.functions.activation.sigmoid
import shape.komputation.functions.hadamard
import shape.komputation.matrix.DoubleMatrix

class SigmoidLayer(name : String? = null) : ActivationLayer(name) {

    private var forwardEntries : DoubleArray = DoubleArray(0)

    private var differentiation : DoubleArray? = null

    override fun forward(input : DoubleMatrix): DoubleMatrix {

        val result = DoubleMatrix(input.numberRows, input.numberColumns, sigmoid(input.entries))

        this.forwardEntries = result.entries

        this.differentiation = null

        return result

    }

    /*
        input = pre-activation
        output = activation

        d activation / d pre-activation = activation * (1 - activation)
     */
    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        if (this.differentiation == null) {

            this.differentiation = differentiateSigmoid(this.forwardEntries)

        }

        return DoubleMatrix(chain.numberRows, chain.numberColumns, hadamard(chain.entries, differentiation!!))

    }

}