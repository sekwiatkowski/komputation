package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.functions.repeatColumn
import shape.komputation.cpu.functions.sumRows
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix

class CpuColumnRepetitionLayer internal constructor(name : String? = null, private val numberRows : Int, private val numberColumns : Int) : BaseCpuForwardLayer(name) {

    private val forwardEntries = FloatArray(this.numberRows * this.numberColumns)
    private val backwardEntries = FloatArray(this.numberRows)

    override fun forward(input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        repeatColumn(input.entries, this.forwardEntries, this.numberColumns)

        return FloatMatrix(this.numberRows, this.numberColumns, this.forwardEntries)

    }

    override fun backward(chain : FloatMatrix): FloatMatrix {

        val chainEntries = chain.entries
        val numberChainRows = chain.numberRows
        val numberChainColumns = chain.numberColumns

        sumRows(numberChainRows, numberChainColumns, chainEntries, this.backwardEntries)

        return FloatMatrix(numberChainRows, 1, this.backwardEntries)

    }

}