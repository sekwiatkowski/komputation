package shape.komputation.layers.combination

import shape.komputation.functions.add
import shape.komputation.layers.CombinationLayer
import shape.komputation.matrix.DoubleMatrix

class AdditionCombination(name : String?) : CombinationLayer(name) {

    override fun forward(first: DoubleMatrix, second: DoubleMatrix) =

        DoubleMatrix(first.numberRows, first.numberColumns, add(first.entries, second.entries))

    // d (x + y) / d x = 1
    override fun backwardFirst(chain: DoubleMatrix) =

        chain

    // d (x + y) / d y = 1
    override fun backwardSecond(chain: DoubleMatrix) =

        chain

}