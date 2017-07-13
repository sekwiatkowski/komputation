package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.layers.BaseForwardLayer
import shape.komputation.functions.negate
import shape.komputation.functions.subtract
import shape.komputation.matrix.DoubleMatrix

class CpuCounterProbabilityLayer internal constructor(
    name : String?,
    dimension : Int) : BaseForwardLayer(name) {

    private val one = DoubleArray(dimension) { 1.0 }

    override fun forward(input: DoubleMatrix, isTraining : Boolean) =

        DoubleMatrix(input.numberRows, input.numberColumns, subtract(one, input.entries))

    override fun backward(chain: DoubleMatrix) =

        DoubleMatrix(chain.numberRows, chain.numberColumns, negate(chain.entries))

}