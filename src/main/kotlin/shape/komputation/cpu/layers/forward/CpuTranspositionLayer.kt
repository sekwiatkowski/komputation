package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.functions.transpose
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix

class CpuTranspositionLayer internal constructor(
    name : String? = null,
    private val numberRows : Int,
    private val numberColumns : Int) : BaseCpuForwardLayer(name)  {

    private val numberEntries = this.numberRows * this.numberColumns
    private val forwardEntries = FloatArray(this.numberEntries)
    private val backwardEntries = FloatArray(this.numberEntries)

    override fun forward(withinBatch : Int, input: FloatMatrix, isTraining : Boolean): FloatMatrix {

        transpose(this.numberRows, this.numberColumns, input.entries, this.forwardEntries)

        return FloatMatrix(this.numberColumns, this.numberRows, this.forwardEntries)

    }

    override fun backward(withinBatch : Int, chain: FloatMatrix): FloatMatrix {

        transpose(this.numberRows, this.numberColumns, chain.entries, this.backwardEntries)

        return FloatMatrix(this.numberColumns, this.numberRows, this.backwardEntries)

    }

}