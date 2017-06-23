package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.backwardTanh
import shape.komputation.functions.activation.tanh
import shape.komputation.layers.ContinuationLayer
import shape.komputation.matrix.DoubleMatrix

class TanhLayer(name : String? = null) : ActivationLayer(name) {

    private var forwardResult : DoubleMatrix? = null

    override fun forward(input : DoubleMatrix) : DoubleMatrix {

        this.forwardResult = DoubleMatrix(input.numberRows, input.numberColumns, tanh(input.entries))

        return this.forwardResult!!

    }

    /*
        Note that each pre-activation effects all nodes.
        For i == j: prediction (1 - prediction)
        for i != j: -(prediction_i * prediction_j)
     */
    override fun backward(chain : DoubleMatrix): DoubleMatrix {

        val forwardResult = this.forwardResult!!
        val forwardEntries = forwardResult.entries
        val numberForwardRows = forwardResult.numberRows
        val numberForwardColumns = forwardResult.numberColumns

        val chainEntries = chain.entries

        val gradient = backwardTanh(forwardEntries, chainEntries)

        return DoubleMatrix(numberForwardRows, numberForwardColumns, gradient)

    }

}