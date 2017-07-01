package shape.komputation.layers.feedforward

import shape.komputation.functions.negate
import shape.komputation.functions.subtract
import shape.komputation.layers.ContinuationLayer
import shape.komputation.matrix.DoubleMatrix

class CounterProbabilityLayer(
    name : String?,
    dimension : Int) : ContinuationLayer(name) {

    private val one = DoubleArray(dimension) { 1.0 }

    override fun forward(input: DoubleMatrix) =

        DoubleMatrix(input.numberRows, input.numberColumns, subtract(one, input.entries))

    override fun backward(chain: DoubleMatrix) =

        DoubleMatrix(chain.numberRows, chain.numberColumns, negate(chain.entries))

}