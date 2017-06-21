package shape.komputation.layers.feedforward.convolution

import shape.komputation.functions.findMaxIndicesInRows
import shape.komputation.functions.selectEntries
import shape.komputation.layers.ContinuationLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleColumnVector

class MaxPoolingLayer(name : String? = null) : ContinuationLayer(name) {

    var input : DoubleMatrix? = null
    var maxRowIndices : IntArray? = null

    override fun forward(input : DoubleMatrix) : DoubleMatrix {

        this.input = input

        val numberRows = input.numberRows
        val numberColumns = input.numberColumns

        val maxRowIndices = findMaxIndicesInRows(input.entries, numberRows, numberColumns)
        this.maxRowIndices = maxRowIndices

        val maxPooled = selectEntries(input.entries, maxRowIndices)

        return doubleColumnVector(*maxPooled)

    }

    override fun backward(chain : DoubleMatrix): DoubleMatrix {

        val input = this.input!!
        val numberInputRows = input.numberRows
        val numberInputColumns = input.numberColumns

        val chainEntries = chain.entries

        val maxRowIndices = this.maxRowIndices!!

        val gradient = DoubleArray(numberInputRows * numberInputColumns)

        for (indexRow in 0..numberInputRows - 1) {

            gradient[maxRowIndices[indexRow]] = chainEntries[indexRow]

        }

        return DoubleMatrix(numberInputRows, numberInputColumns, gradient)

    }

}