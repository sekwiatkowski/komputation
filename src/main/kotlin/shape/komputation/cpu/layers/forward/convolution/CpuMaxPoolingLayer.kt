package shape.komputation.cpu.layers.forward.convolution

import shape.komputation.cpu.functions.findMaxIndicesInRows
import shape.komputation.cpu.functions.selectEntries
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.floatColumnVector

class CpuMaxPoolingLayer internal constructor(name : String? = null, private val numberRows : Int) : BaseCpuForwardLayer(name) {

    private var numberColumns = -1
    private val maxRowIndices = IntArray(this.numberRows)
    private val forwardEntries = FloatArray(this.numberRows)

    override fun forward(withinBatch : Int, input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        this.numberColumns = input.numberColumns

        findMaxIndicesInRows(input.entries, this.numberRows, this.numberColumns, this.maxRowIndices)

        selectEntries(input.entries, this.maxRowIndices, this.forwardEntries, this.numberRows)

        return floatColumnVector(*this.forwardEntries)

    }

    override fun backward(withinBatch : Int, chain : FloatMatrix): FloatMatrix {

        val chainEntries = chain.entries

        val gradient = FloatArray(this.numberRows * this.numberColumns)

        for (indexRow in 0..this.numberRows - 1) {

            gradient[this.maxRowIndices[indexRow]] = chainEntries[indexRow]

        }

        return FloatMatrix(this.numberRows, this.numberColumns, gradient)

    }

}