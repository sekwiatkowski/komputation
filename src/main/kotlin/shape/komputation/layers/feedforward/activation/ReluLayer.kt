package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.backwardRelu
import shape.komputation.functions.activation.relu
import shape.komputation.layers.ContinuationLayer
import shape.komputation.matrix.DoubleMatrix

class ReluLayer(name : String? = null) : ActivationLayer(name) {

    private var forwardEntries : DoubleArray = DoubleArray(0)
    private var numberForwardRows = -1
    private var numberForwardColumns = -1

    override fun forward(input : DoubleMatrix): DoubleMatrix {

        val result = DoubleMatrix(input.numberRows, input.numberColumns, relu(input.entries))

        this.forwardEntries = result.entries
        this.numberForwardRows = result.numberRows
        this.numberForwardColumns = result.numberColumns

        return result

    }

    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        val chainEntries = chain.entries

        val backwardEntries = backwardRelu(this.forwardEntries, chainEntries)

        return DoubleMatrix(this.numberForwardRows, this.numberForwardColumns, backwardEntries)

    }

}