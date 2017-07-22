package shape.komputation.cpu.layers.forward.convolution

import shape.komputation.cpu.functions.findMaxIndicesInRows
import shape.komputation.cpu.functions.selectEntries
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.floatColumnVector

class CpuMaxPoolingLayer internal constructor(name : String? = null) : BaseCpuForwardLayer(name) {

    private var numberInputRows = -1
    private var numberInputColumns = -1
    private var maxRowIndices = IntArray(0)

    override fun forward(input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        this.numberInputRows = input.numberRows
        this.numberInputColumns = input.numberColumns

        val maxRowIndices = findMaxIndicesInRows(input.entries, this.numberInputRows, this.numberInputColumns)
        this.maxRowIndices = maxRowIndices

        val maxPooled = selectEntries(input.entries, maxRowIndices)

        return floatColumnVector(*maxPooled)

    }

    override fun backward(chain : FloatMatrix): FloatMatrix {

        val chainEntries = chain.entries

        val gradient = FloatArray(this.numberInputRows * this.numberInputColumns)

        for (indexRow in 0..this.numberInputRows - 1) {

            gradient[this.maxRowIndices[indexRow]] = chainEntries[indexRow]

        }

        return FloatMatrix(this.numberInputRows, this.numberInputColumns, gradient)

    }

}