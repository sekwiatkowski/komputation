package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.functions.activation.backwardVectorSoftmax
import shape.komputation.cpu.functions.activation.vectorSoftmax
import shape.komputation.matrix.DoubleMatrix

class CpuSoftmaxVectorLayer internal constructor (name : String? = null) : BaseCpuActivationLayer(name) {

    private var forwardEntries : DoubleArray = DoubleArray(0)

    override fun forward(input : DoubleMatrix, isTraining : Boolean) : DoubleMatrix {

        val result = DoubleMatrix(input.numberRows, input.numberColumns, vectorSoftmax(input.entries))

        this.forwardEntries = result.entries

        return result

    }

    /*
        Note that each pre-activation effects all nodes.
        For i == j: prediction (1 - prediction)
        for i != j: -(prediction_i * prediction_j)
     */
    override fun backward(chain : DoubleMatrix): DoubleMatrix {

        val chainEntries = chain.entries

        return DoubleMatrix(chain.numberRows, chain.numberColumns, backwardVectorSoftmax(this.forwardEntries, chainEntries))

    }

}