package shape.komputation.layers.entry

import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix

abstract class EntryPoint(private val name : String? = null) {

    abstract fun forward(input: Matrix) : DoubleMatrix

    abstract fun backward(chain : DoubleMatrix) : DoubleMatrix

}