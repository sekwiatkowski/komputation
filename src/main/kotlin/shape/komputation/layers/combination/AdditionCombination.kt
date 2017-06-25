package shape.komputation.layers.combination

import shape.komputation.functions.add
import shape.komputation.layers.CombinationLayer
import shape.komputation.matrix.DoubleMatrix

class AdditionCombination(name : String?) : CombinationLayer(name) {

    override fun forward(first: DoubleMatrix, second: DoubleMatrix) =

        DoubleMatrix(first.numberRows, first.numberColumns, add(first.entries, second.entries))

    override fun backwardFirst(chain: DoubleMatrix) =

        chain

    override fun backwardSecond(chain: DoubleMatrix) =

        chain

}