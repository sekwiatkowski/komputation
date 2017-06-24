package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.backwardSigmoid
import shape.komputation.functions.activation.sigmoid
import shape.komputation.layers.ContinuationLayer
import shape.komputation.matrix.DoubleMatrix

class SigmoidLayer(name : String? = null) : ActivationLayer(name) {

    private var forwardEntries : DoubleArray = DoubleArray(0)
    private var numberForwardRows = -1
    private var numberForwardColumns = -1

    override fun forward(input : DoubleMatrix): DoubleMatrix {

        val result = DoubleMatrix(input.numberRows, input.numberColumns, sigmoid(input.entries))

        this.forwardEntries = result.entries
        this.numberForwardRows = result.numberRows
        this.numberForwardColumns = result.numberColumns

        return result

    }

    /*
        input = pre-activation
        output = activation

        d activation / d pre-activation = activation * (1 - activation)
     */
    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        val chainEntries = chain.entries

        val entries = backwardSigmoid(this.forwardEntries, chainEntries)

        return DoubleMatrix(this.numberForwardRows, this.numberForwardColumns, entries)

    }

}