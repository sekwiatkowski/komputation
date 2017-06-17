package shape.komputation.layers.entry

import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix

class InputLayer(name : String? = null) : EntryPoint(name) {

    override fun forward(input: Matrix) =

        input as DoubleMatrix

    override fun backward(chain : DoubleMatrix) =

        chain

}