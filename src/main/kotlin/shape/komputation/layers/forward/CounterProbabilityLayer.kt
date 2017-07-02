package shape.komputation.layers.forward

import shape.komputation.functions.negate
import shape.komputation.functions.subtract
import shape.komputation.layers.ForwardLayer
import shape.komputation.matrix.DoubleMatrix

class CounterProbabilityLayer(
    name : String?,
    dimension : Int) : ForwardLayer(name) {

    private val one = DoubleArray(dimension) { 1.0 }

    override fun forward(input: DoubleMatrix, isTraining : Boolean) =

        DoubleMatrix(input.numberRows, input.numberColumns, subtract(one, input.entries))

    override fun backward(chain: DoubleMatrix) =

        DoubleMatrix(chain.numberRows, chain.numberColumns, negate(chain.entries))

}