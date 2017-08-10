package shape.komputation.cpu.layers.entry

import shape.komputation.cpu.layers.BaseCpuEntryPoint
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.Matrix

class CpuInputLayer internal constructor(
    name : String? = null,
    numberInputRows: Int,
    numberInputColumns: Int) : BaseCpuEntryPoint(name) {

    override val numberOutputRows = numberInputRows
    override val numberOutputColumns = numberInputColumns
    override var forwardResult = FloatArray(0)

    override fun forward(input: Matrix): FloatArray {

        this.forwardResult = (input as FloatMatrix).entries

        return this.forwardResult

    }

    override fun backward(chain : FloatArray) =

        chain

}