package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.backwardSigmoid
import shape.komputation.functions.activation.sigmoid
import shape.komputation.matrix.DoubleMatrix

class SigmoidLayer(name : String? = null) : ActivationLayer(name) {

    private var forwardResult : DoubleMatrix? = null

    override fun forward(input : DoubleMatrix): DoubleMatrix {

        this.forwardResult = DoubleMatrix(input.numberRows, input.numberColumns, sigmoid(input.entries))

        return this.forwardResult!!

    }

    /*
        input = pre-activation
        output = activation

        d activation / d pre-activation = activation * (1 - activation)
     */
    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        val chainEntries = chain.entries

        val forwardResult = this.forwardResult!!
        val forwardResultEntries = forwardResult.entries

        val entries = backwardSigmoid(forwardResultEntries, chainEntries)

        return DoubleMatrix(forwardResult.numberRows, forwardResult.numberColumns, entries)

    }

}