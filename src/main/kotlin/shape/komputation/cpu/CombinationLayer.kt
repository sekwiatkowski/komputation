package shape.komputation.cpu

import shape.komputation.matrix.DoubleMatrix

abstract class CombinationLayer(private val name: String?) {

    abstract fun forward(first: DoubleMatrix, second: DoubleMatrix) : DoubleMatrix

    abstract fun backwardFirst(chain : DoubleMatrix) : DoubleMatrix

    abstract fun backwardSecond(chain : DoubleMatrix) : DoubleMatrix

}