package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.differentiateTanh
import shape.komputation.functions.activation.tanh
import shape.komputation.functions.hadamard
import shape.komputation.matrix.DoubleMatrix

class TanhLayer(name : String? = null) : ActivationLayer(name) {

    private var forwardEntries : DoubleArray = DoubleArray(0)
    private var numberForwardRows = -1
    private var numberForwardColumns = -1

    private var differentiation : DoubleArray? = null

    override fun forward(input : DoubleMatrix) : DoubleMatrix {

        val result = DoubleMatrix(input.numberRows, input.numberColumns, tanh(input.entries))

        this.forwardEntries = result.entries
        this.numberForwardRows = result.numberRows
        this.numberForwardColumns = result.numberColumns

        this.differentiation = null

        return result
    }

    /*
        Note that each pre-activation effects all nodes.
        For i == j: prediction (1 - prediction)
        for i != j: -(prediction_i * prediction_j)
     */
    override fun backward(chain : DoubleMatrix): DoubleMatrix {

        this.differentiation = differentiateTanh(this.forwardEntries)

        return DoubleMatrix(this.numberForwardRows, this.numberForwardColumns, hadamard(chain.entries, differentiation!!))

    }

}