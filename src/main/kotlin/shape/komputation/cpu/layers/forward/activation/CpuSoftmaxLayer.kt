package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.functions.activation.backwardColumnWiseSoftmax
import shape.komputation.cpu.functions.activation.columnWiseSoftmax
import shape.komputation.matrix.FloatMatrix

class CpuSoftmaxLayer internal constructor(name : String? = null) : BaseCpuActivationLayer(name) {

    private var forwardEntries = FloatArray(0)

    override fun forward(input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        val numberRows = input.numberRows
        val numberColumns = input.numberColumns

        val result = FloatMatrix(numberRows, numberColumns, columnWiseSoftmax(input.entries, numberRows, numberColumns))

        this.forwardEntries = result.entries

        return result

    }

    /*
        Note that each pre-activation effects all nodes.
        For i == j: prediction (1 - prediction)
        for i != j: -(prediction_i * prediction_j)
     */
    override fun backward(chain : FloatMatrix): FloatMatrix {

        val chainEntries = chain.entries

        val numberRows = chain.numberRows
        val numberColumns = chain.numberColumns

        val gradient = backwardColumnWiseSoftmax(numberRows, numberColumns, this.forwardEntries, chainEntries)

        return FloatMatrix(numberRows, numberColumns, gradient)

    }

}