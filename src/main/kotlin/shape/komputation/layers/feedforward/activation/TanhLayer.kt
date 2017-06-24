package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.backwardTanh
import shape.komputation.functions.activation.tanh
import shape.komputation.layers.ContinuationLayer
import shape.komputation.matrix.DoubleMatrix

class TanhLayer(name : String? = null) : ActivationLayer(name) {

    private var forwardEntries : DoubleArray = DoubleArray(0)
    private var numberForwardRows = -1
    private var numberForwardColumns = -1

    override fun forward(input : DoubleMatrix) : DoubleMatrix {

        val result = DoubleMatrix(input.numberRows, input.numberColumns, tanh(input.entries))

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

        val gradient = backwardTanh(this.forwardEntries, chainEntries)

        return DoubleMatrix(this.numberForwardRows, this.numberForwardColumns, gradient)

    }

}