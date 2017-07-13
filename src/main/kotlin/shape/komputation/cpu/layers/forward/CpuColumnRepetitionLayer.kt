package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.layers.BaseForwardLayer
import shape.komputation.functions.repeatColumn
import shape.komputation.functions.sumRows
import shape.komputation.matrix.DoubleMatrix

class CpuColumnRepetitionLayer internal constructor(name : String? = null, private val n : Int) : BaseForwardLayer(name) {

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