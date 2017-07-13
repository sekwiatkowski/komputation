package shape.komputation.layers.forward

import shape.komputation.functions.negate
import shape.komputation.functions.subtract
import shape.komputation.layers.BaseForwardLayer
import shape.komputation.layers.CpuForwardLayerInstruction
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

class CounterProbabilityLayer(
    private val name : String?,
    private val dimension: Int) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuCounterProbabilityLayer(this.name, this.dimension)

}


fun counterProbabilityLayer(dimension: Int) =

    counterProbabilityLayer(null, dimension)

fun counterProbabilityLayer(name : String?, dimension: Int) =

    CounterProbabilityLayer(name, dimension)