package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.functions.negate
import shape.komputation.cpu.functions.subtract
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.layers.Resourceful

class CpuCounterProbabilityLayer internal constructor(
    name : String?,
    private val numberRows: Int,
    private val numberColumns : Int) : BaseCpuForwardLayer(name), Resourceful {

    private val numberEntries = this.numberRows * this.numberColumns

    private val one = FloatArray(this.numberEntries) { 1.0f }

    override val numberOutputRows = this.numberRows
    override val numberOutputColumns = this.numberColumns
    override var forwardResult = FloatArray(0)

    override val numberInputRows = this.numberRows
    override val numberInputColumns = this.numberColumns
    override var backwardResult = FloatArray(0)

    override fun acquire(maximumBatchSize: Int) {

        this.forwardResult = FloatArray(this.numberEntries)
        this.backwardResult = FloatArray(this.numberEntries)

    }

    override fun release() {

    }

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean): FloatArray {

        subtract(this.one, input, this.numberEntries, this.forwardResult)

        return this.forwardResult

    }

    override fun backward(withinBatch : Int, chain: FloatArray): FloatArray {

        negate(chain, this.backwardResult, this.numberEntries)

        return this.backwardResult

    }

}