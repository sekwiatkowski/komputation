package shape.komputation.layers.forward.convolution

import shape.komputation.functions.findMaxIndicesInRows
import shape.komputation.functions.selectEntries
import shape.komputation.layers.ForwardLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleColumnVector

class MaxPoolingLayer(name : String? = null) : ForwardLayer(name) {

    private var numberInputRows = -1
    private var numberInputColumns = -1
    private var maxRowIndices = IntArray(0)

    override fun forward(input : DoubleMatrix, isTraining : Boolean) : DoubleMatrix {

        this.numberInputRows = input.numberRows
        this.numberInputColumns = input.numberColumns

        val maxRowIndices = findMaxIndicesInRows(input.entries, this.numberInputRows, this.numberInputColumns)
        this.maxRowIndices = maxRowIndices

        val maxPooled = selectEntries(input.entries, maxRowIndices)

        return doubleColumnVector(*maxPooled)

    }

    override fun backward(chain : DoubleMatrix): DoubleMatrix {

        val chainEntries = chain.entries

        val gradient = DoubleArray(this.numberInputRows * this.numberInputColumns)

        for (indexRow in 0..this.numberInputRows - 1) {

            gradient[this.maxRowIndices[indexRow]] = chainEntries[indexRow]

        }

        return DoubleMatrix(this.numberInputRows, this.numberInputColumns, gradient)

    }

}