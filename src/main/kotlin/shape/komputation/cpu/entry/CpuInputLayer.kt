package shape.komputation.cpu.entry

import shape.komputation.cpu.BaseCpuEntryPoint
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix

class CpuInputLayer internal constructor(name : String? = null) : BaseCpuEntryPoint(name) {

    override fun forward(input: Matrix) =

        input as DoubleMatrix

    override fun backward(chain : DoubleMatrix) =

        chain

}