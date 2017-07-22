package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.functions.negate
import shape.komputation.cpu.functions.subtract
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix

class CpuCounterProbabilityLayer internal constructor(
    name : String?,
    dimension : Int) : BaseCpuForwardLayer(name) {

    private val one = FloatArray(dimension) { 1.0f }

    override fun forward(input: FloatMatrix, isTraining : Boolean) =

        FloatMatrix(input.numberRows, input.numberColumns, subtract(one, input.entries))

    override fun backward(chain: FloatMatrix) =

        FloatMatrix(chain.numberRows, chain.numberColumns, negate(chain.entries))

}