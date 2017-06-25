package shape.komputation.layers.combination

import shape.komputation.functions.negate
import shape.komputation.functions.subtract
import shape.komputation.layers.CombinationLayer
import shape.komputation.matrix.DoubleMatrix

// a - b
class SubtractionCombination(val name : String?) : CombinationLayer(name) {

    override fun forward(first: DoubleMatrix, second: DoubleMatrix) =

        DoubleMatrix(first.numberRows, first.numberColumns, subtract(first.entries, second.entries))

    override fun backwardFirst(chain: DoubleMatrix) =

        chain

    override fun backwardSecond(chain: DoubleMatrix) =

        DoubleMatrix(chain.numberRows, chain.numberColumns, negate(chain.entries))

}