package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.functions.negate
import shape.komputation.cpu.functions.subtract
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix

class CpuCounterProbabilityLayer internal constructor(
    name : String?,
    private val numberEntries: Int) : BaseCpuForwardLayer(name) {

    private val one = FloatArray(this.numberEntries) { 1.0f }
    private val forwardEntries = FloatArray(this.numberEntries)
    private val backwardEntries = FloatArray(this.numberEntries)

    override fun forward(input: FloatMatrix, isTraining : Boolean): FloatMatrix {

        subtract(this.one, input.entries, this.forwardEntries, this.numberEntries)

        return FloatMatrix(input.numberRows, input.numberColumns, this.forwardEntries)

    }

    override fun backward(chain: FloatMatrix): FloatMatrix {

        negate(chain.entries, this.backwardEntries, this.numberEntries)

        return FloatMatrix(chain.numberRows, chain.numberColumns, this.backwardEntries)

    }

}