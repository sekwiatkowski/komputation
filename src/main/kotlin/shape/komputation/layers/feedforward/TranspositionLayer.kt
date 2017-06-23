package shape.komputation.layers.feedforward

import shape.komputation.functions.transpose
import shape.komputation.layers.ContinuationLayer
import shape.komputation.matrix.DoubleMatrix

class TranspositionLayer(name : String? = null) : ContinuationLayer(name)  {

    override fun forward(input: DoubleMatrix) =

        DoubleMatrix(input.numberColumns, input.numberRows, transpose(input.numberRows, input.numberColumns, input.entries))

    override fun backward(chain: DoubleMatrix) =

        DoubleMatrix(chain.numberColumns, chain.numberRows, transpose(chain.numberRows, chain.numberColumns, chain.entries))

}