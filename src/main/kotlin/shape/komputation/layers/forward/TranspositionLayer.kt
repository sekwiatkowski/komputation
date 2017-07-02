package shape.komputation.layers.forward

import shape.komputation.functions.transpose
import shape.komputation.layers.ForwardLayer
import shape.komputation.matrix.DoubleMatrix

class TranspositionLayer(name : String? = null) : ForwardLayer(name)  {

    override fun forward(input: DoubleMatrix, isTraining : Boolean) =

        DoubleMatrix(input.numberColumns, input.numberRows, transpose(input.numberRows, input.numberColumns, input.entries))

    override fun backward(chain: DoubleMatrix) =

        DoubleMatrix(chain.numberColumns, chain.numberRows, transpose(chain.numberRows, chain.numberColumns, chain.entries))

}