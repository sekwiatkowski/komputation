package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.backwardRelu
import shape.komputation.functions.activation.relu
import shape.komputation.layers.ContinuationLayer
import shape.komputation.matrix.DoubleMatrix

class ReluLayer(name : String? = null) : ContinuationLayer(name), ActivationLayer {

    private var forwardResult : DoubleMatrix? = null

    override fun forward(input : DoubleMatrix): DoubleMatrix {

        this.forwardResult = DoubleMatrix(input.numberRows, input.numberColumns, relu(input.entries))

        return this.forwardResult!!

    }

    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        val forwardResult = this.forwardResult!!
        val forwardEntries = forwardResult.entries

        val chainEntries = chain.entries

        val backwardEntries = backwardRelu(forwardEntries, chainEntries)

        return DoubleMatrix(forwardResult.numberRows, forwardResult.numberColumns, backwardEntries)

    }

}