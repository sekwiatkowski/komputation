package shape.komputation.layers

import shape.komputation.matrix.DoubleMatrix

abstract class ContinuationLayer(private val name: String?) {

    abstract fun forward(input: DoubleMatrix) : DoubleMatrix

    abstract fun backward(chain : DoubleMatrix) : DoubleMatrix

}