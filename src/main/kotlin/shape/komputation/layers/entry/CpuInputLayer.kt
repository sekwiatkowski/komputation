package shape.komputation.layers.entry

import shape.komputation.layers.BaseEntryPoint
import shape.komputation.layers.CpuEntryPointInstruction
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix

class CpuInputLayer internal constructor(name : String? = null) : BaseEntryPoint(name) {

    override fun forward(input: Matrix) =

        input as DoubleMatrix

    override fun backward(chain : DoubleMatrix) =

        chain

}

class InputLayer(private val name : String? = null) : CpuEntryPointInstruction {

    override fun buildForCpu() =

        CpuInputLayer(this.name)


}

fun inputLayer(name : String? = null) = InputLayer(name)