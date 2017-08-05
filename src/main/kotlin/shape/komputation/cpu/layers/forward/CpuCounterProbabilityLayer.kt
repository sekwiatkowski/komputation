package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.functions.negate
import shape.komputation.cpu.functions.subtract
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix

class CpuCounterProbabilityLayer internal constructor(
    name : String?,
    private val numberRows: Int,
    private val numberColumns : Int) : BaseCpuForwardLayer(name) {

    private val numberEntries = this.numberRows * this.numberColumns

    private val one = FloatArray(this.numberEntries) { 1.0f }
    private val forwardEntries = FloatArray(this.numberEntries)
    private val backwardEntries = FloatArray(this.numberEntries)

    override fun forward(withinBatch : Int, input: FloatMatrix, isTraining : Boolean): FloatMatrix {

        subtract(this.one, input.entries, this.forwardEntries, this.numberEntries)

        return FloatMatrix(this.numberRows, this.numberColumns, this.forwardEntries)

    }

    override fun backward(withinBatch : Int, chain: FloatMatrix): FloatMatrix {

        negate(chain.entries, this.backwardEntries, this.numberEntries)

        return FloatMatrix(this.numberRows, this.numberColumns, this.backwardEntries)

    }

}