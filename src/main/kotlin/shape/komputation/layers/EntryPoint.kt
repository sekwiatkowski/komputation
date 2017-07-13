package shape.komputation.layers

import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix

interface EntryPoint {

    fun forward(input: Matrix) : DoubleMatrix

    fun backward(chain : DoubleMatrix) : DoubleMatrix

}