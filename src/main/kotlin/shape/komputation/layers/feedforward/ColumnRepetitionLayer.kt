package shape.komputation.layers.feedforward

import shape.komputation.functions.repeatColumn
import shape.komputation.functions.sumRows
import shape.komputation.layers.ContinuationLayer
import shape.komputation.matrix.DoubleMatrix

class ColumnRepetitionLayer(name : String? = null, private val n : Int) : ContinuationLayer(name) {

    override fun forward(input : DoubleMatrix) : DoubleMatrix {

        val inputEntries = input.entries
        val inputSize = inputEntries.size

        return DoubleMatrix(inputSize, n, repeatColumn(inputEntries, n))

    }

    override fun backward(chain : DoubleMatrix): DoubleMatrix {

        return DoubleMatrix(chain.numberRows, 1, sumRows(chain.entries, chain.numberRows, chain.numberColumns))

    }

}