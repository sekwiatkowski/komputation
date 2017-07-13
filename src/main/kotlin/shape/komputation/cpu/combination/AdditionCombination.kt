package shape.komputation.cpu.combination

import shape.komputation.cpu.CombinationLayer
import shape.komputation.functions.add
import shape.komputation.matrix.DoubleMatrix

/*
   Ex:
   f(x) = (g(x)+h(x))^2 = g(x)^2 + 2*g(x)*h(x) + h(x)^2
   d f(x) / g(x) = 2*g(x) + 2*h(x)
   d f(x) / h(x) = 2*h(x) + 2*g(x)
   chain = d (g(x)+h(x))^2 / d g(x)+h(x) = 2 * (g(x)+h(x)) = 2*g(x) + 2*h(x)
   chain * d g(x)+h(x) / d g(x) = chain
   chain * d g(x)+h(x) / d h(x) = chain
*/
class AdditionCombination internal constructor(name : String? = null) : CombinationLayer(name) {

    override fun forward(first: DoubleMatrix, second: DoubleMatrix) =

        DoubleMatrix(first.numberRows, first.numberColumns, add(first.entries, second.entries))

    // d (x + y) / d x = 1
    override fun backwardFirst(chain: DoubleMatrix) =

        chain

    // d (x + y) / d y = 1
    override fun backwardSecond(chain: DoubleMatrix) =

        chain

}

fun additionCombination(name : String? = null) = AdditionCombination(name)