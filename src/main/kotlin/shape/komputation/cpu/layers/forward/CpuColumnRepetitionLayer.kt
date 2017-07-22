package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.functions.repeatColumn
import shape.komputation.cpu.functions.sumRows
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix

class CpuColumnRepetitionLayer internal constructor(name : String? = null, private val n : Int) : BaseCpuForwardLayer(name) {

    override fun forward(input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        val inputEntries = input.entries
        val inputSize = inputEntries.size

        return FloatMatrix(inputSize, n, repeatColumn(inputEntries, n))

    }

    override fun backward(chain : FloatMatrix): FloatMatrix {

        val chainEntries = chain.entries
        val numberChainRows = chain.numberRows
        val numberChainColumns = chain.numberColumns

        val result = sumRows(chainEntries, numberChainRows, numberChainColumns)

        return FloatMatrix(numberChainRows, 1, result)

    }

}