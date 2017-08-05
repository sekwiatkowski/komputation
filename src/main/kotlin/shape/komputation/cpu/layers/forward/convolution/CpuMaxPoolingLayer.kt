package shape.komputation.cpu.layers.forward.convolution

import shape.komputation.cpu.functions.findMaxIndicesInRows
import shape.komputation.cpu.functions.selectEntries
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.floatColumnVector
import java.util.*

class CpuMaxPoolingLayer internal constructor(name : String? = null, private val numberInputRows : Int, private val numberInputColumns : Int) : BaseCpuForwardLayer(name) {

    private val maxRowIndices = IntArray(this.numberInputRows)
    private val forwardEntries = FloatArray(this.numberInputRows)

    private val numberInputEntries = this.numberInputRows * this.numberInputColumns

    override fun forward(withinBatch : Int, input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        findMaxIndicesInRows(input.entries, this.numberInputRows, this.numberInputColumns, this.maxRowIndices)

        selectEntries(input.entries, this.maxRowIndices, this.forwardEntries, this.numberInputRows)

        return floatColumnVector(*this.forwardEntries)

    }

    private val backwardEntries = FloatArray(this.numberInputEntries)

    override fun backward(withinBatch : Int, chain : FloatMatrix): FloatMatrix {

        val chainEntries = chain.entries

        Arrays.fill(this.backwardEntries, 0f)

        for (indexRow in 0..this.numberInputRows - 1) {

            this.backwardEntries[this.maxRowIndices[indexRow]] = chainEntries[indexRow]

        }

        return FloatMatrix(this.numberInputRows, this.numberInputColumns, this.backwardEntries)

    }

}