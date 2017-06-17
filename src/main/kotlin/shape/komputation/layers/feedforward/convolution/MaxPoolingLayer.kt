package shape.komputation.layers.feedforward.convolution

import shape.komputation.functions.findMaxIndicesInRows
import shape.komputation.functions.selectEntries
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleRowVector

/*
    Example:
    The input is a 20*100 matrix (filter size of 3, 22 words, 100 filters)
    The output is a 100D row vector
 */

class MaxPoolingLayer(name : String? = null) : FeedForwardLayer(name) {

    var input : DoubleMatrix? = null
    var maxRowIndices : IntArray? = null

    override fun forward(input : DoubleMatrix) : DoubleMatrix {

        this.input = input

        val numberRows = input.numberRows
        val numberColumns = input.numberColumns

        val maxRowIndices = findMaxIndicesInRows(input.entries, numberRows, numberColumns)
        this.maxRowIndices = maxRowIndices

        val maxPooled = selectEntries(input.entries, maxRowIndices)

        return doubleRowVector(*maxPooled)

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