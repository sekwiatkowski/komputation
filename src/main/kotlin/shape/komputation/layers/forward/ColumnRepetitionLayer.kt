package shape.komputation.layers.forward

import shape.komputation.functions.repeatColumn
import shape.komputation.functions.sumRows
import shape.komputation.layers.ForwardLayer
import shape.komputation.matrix.DoubleMatrix

class ColumnRepetitionLayer internal constructor(name : String? = null, private val n : Int) : ForwardLayer(name) {

    override fun forward(input : DoubleMatrix, isTraining : Boolean) : DoubleMatrix {

        val inputEntries = input.entries
        val inputSize = inputEntries.size

        return DoubleMatrix(inputSize, n, repeatColumn(inputEntries, n))

    }

    override fun backward(chain : DoubleMatrix): DoubleMatrix {

        val chainEntries = chain.entries
        val numberChainRows = chain.numberRows
        val numberChainColumns = chain.numberColumns

        val result = sumRows(chainEntries, numberChainRows, numberChainColumns)

        return DoubleMatrix(numberChainRows, 1, result)

    }

}

fun columnRepetitionLayer(n : Int) =

    ColumnRepetitionLayer(null, n)

fun columnRepetitionLayer(name : String? = null, n : Int) =

    ColumnRepetitionLayer(name, n)