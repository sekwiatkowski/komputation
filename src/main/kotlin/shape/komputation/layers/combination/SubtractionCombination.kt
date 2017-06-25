package shape.komputation.layers.combination

import shape.komputation.functions.negate
import shape.komputation.functions.subtract
import shape.komputation.layers.CombinationLayer
import shape.komputation.matrix.DoubleMatrix

class SubtractionCombination(val name : String?) : CombinationLayer(name) {

    override fun forward(first: DoubleMatrix, second: DoubleMatrix) =

        DoubleMatrix(first.numberRows, first.numberColumns, subtract(first.entries, second.entries))

    // d (x - y) / d x = 1
    override fun backwardFirst(chain: DoubleMatrix) =

        chain

    // d (x - y) / d y = -1
    override fun backwardSecond(chain: DoubleMatrix) =

        DoubleMatrix(chain.numberRows, chain.numberColumns, negate(chain.entries))

}