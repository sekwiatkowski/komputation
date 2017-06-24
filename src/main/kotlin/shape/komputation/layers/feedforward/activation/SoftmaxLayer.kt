package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.backwardColumnWiseSoftmax
import shape.komputation.functions.activation.columnWiseSoftmax
import shape.komputation.matrix.DoubleMatrix

class SoftmaxLayer(name : String? = null) : ActivationLayer(name) {

    private var forwardEntries : DoubleArray = DoubleArray(0)
    private var numberForwardRows = -1
    private var numberForwardColumns = -1

    override fun forward(input : DoubleMatrix) : DoubleMatrix {

        val numberRows = input.numberRows
        val numberColumns = input.numberColumns

        val result = DoubleMatrix(numberRows, numberColumns, columnWiseSoftmax(input.entries, numberRows, numberColumns))

        this.forwardEntries = result.entries
        this.numberForwardRows = result.numberRows
        this.numberForwardColumns = result.numberColumns

        return result

    }

    /*
        Note that each pre-activation effects all nodes.
        For i == j: prediction (1 - prediction)
        for i != j: -(prediction_i * prediction_j)
     */
    override fun backward(chain : DoubleMatrix): DoubleMatrix {

        val chainEntries = chain.entries

        val gradient = backwardColumnWiseSoftmax(this.numberForwardRows, this.numberForwardColumns, this.forwardEntries, chainEntries)

        return DoubleMatrix(this.numberForwardRows, this.numberForwardColumns, gradient)

    }

}