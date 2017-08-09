package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.functions.repeatColumn
import shape.komputation.cpu.functions.sumRows
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.layers.Resourceful

class CpuColumnRepetitionLayer internal constructor(
    name : String? = null,
    override val numberInputRows: Int,
    override val numberOutputColumns: Int) : BaseCpuForwardLayer(name), Resourceful {

    override val numberOutputRows = this.numberInputRows
    override val numberInputColumns = 1

    override var forwardResult = FloatArray(0)
    override var backwardResult = FloatArray(0)

    override fun acquire(maximumBatchSize: Int) {

        this.forwardResult = FloatArray(this.numberInputRows * this.numberOutputColumns)
        this.backwardResult = FloatArray(this.numberInputRows)

    }

    override fun release() {

    }

    override fun forward(withinBatch : Int, numberInputColumns : Int, input : FloatArray, isTraining : Boolean): FloatArray {

        repeatColumn(input, this.numberOutputColumns, this.forwardResult)

        return this.forwardResult

    }

    override fun backward(withinBatch : Int, chain : FloatArray): FloatArray {

        sumRows(this.numberInputRows, this.numberOutputColumns, chain, this.backwardResult)

        return this.backwardResult

    }

}