package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.functions.activation.backwardColumnWiseSoftmax
import shape.komputation.cpu.functions.activation.columnWiseSoftmax
import shape.komputation.matrix.FloatMatrix

class CpuSoftmaxLayer internal constructor(name : String? = null, private val numberRows : Int, private val numberColumns : Int) : BaseCpuActivationLayer(name) {

    private var forwardEntries = FloatArray(this.numberRows * this.numberColumns)
    private var backwardEntries = FloatArray(this.numberRows * this.numberColumns)

    override fun forward(input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        columnWiseSoftmax(input.entries, this.numberRows, this.numberColumns, this.forwardEntries)

        val result = FloatMatrix(this.numberRows, this.numberColumns, this.forwardEntries)

        return result

    }

    /*
        Note that each pre-activation effects all nodes.
        For i == j: prediction (1 - prediction)
        for i != j: -(prediction_i * prediction_j)
     */
    override fun backward(chain : FloatMatrix): FloatMatrix {

        backwardColumnWiseSoftmax(this.numberRows, this.numberColumns, this.forwardEntries, chain.entries, this.backwardEntries)

        return FloatMatrix(this.numberRows, this.numberColumns, this.backwardEntries)

    }

}